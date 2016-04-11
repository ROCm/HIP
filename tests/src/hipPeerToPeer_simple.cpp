/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
// Simple test for memset.
// Also serves as a template for other tests.

#include "hip_runtime.h"
#include "test_common.h"

bool p_memcpyWithPeer = false; // use the peer device for the P2P copy
bool p_mirrorPeers = false; // in addition to mapping current to peer space, map peer to current space.
int  p_peerDevice = -1;  // explicly specify which peer to use, else use p_gpuDevice + 1.

void parseMyArguments(int argc, char *argv[])
{
    int more_argc = HipTest::parseStandardArguments(argc, argv, false);
    // parse args for this test:
    for (int i = 1; i < more_argc; i++) {
        const char *arg = argv[i];

        if (!strcmp(arg, "--memcpyWithPeer")) {
            p_memcpyWithPeer = true;
        } else if (!strcmp(arg, "--mirrorPeers")) {
            p_mirrorPeers = true;
        } else if (!strcmp(arg, "--peerDevice")) {
            if (++i >= argc || !HipTest::parseInt(argv[i], &p_peerDevice)) {
               failed("Bad peerDevice argument"); 
            }
        } else {
            failed("Bad argument '%s'", arg);
        }
    };
};


//---
// Test which enables peer2peer first, then allocates the memory.
void enablePeerFirst()
{
    printf ("\n==testing: %s\n", __func__);

    int deviceCnt;

    HIPCHECK(hipGetDeviceCount(&deviceCnt));

    int currentDevice = p_gpuDevice;
    int peerDevice    = (p_peerDevice == -1) ? ((currentDevice + 1) % deviceCnt) : p_peerDevice;

    printf ("N=%zu  device=%d peerDevice=%d (%d devices total)\n", N, currentDevice, peerDevice, deviceCnt);

    // Must be on a multi-gpu system:
    assert (currentDevice != peerDevice);

    int canAccessPeer;
    HIPCHECK(hipDeviceCanAccessPeer(&canAccessPeer, currentDevice, peerDevice));
    printf ("dev#%d canAccessPeer:#%d=%d\n", currentDevice, peerDevice, canAccessPeer);

    assert(canAccessPeer);

    HIPCHECK (hipSetDevice(currentDevice));
    HIPCHECK(hipDeviceReset());
    HIPCHECK (hipSetDevice(peerDevice));
    HIPCHECK(hipDeviceReset());

    HIPCHECK(hipSetDevice(currentDevice));
    HIPCHECK(hipDeviceEnablePeerAccess(peerDevice, 0));

    if (p_mirrorPeers) {
        int canAccessPeer;
        HIPCHECK(hipDeviceCanAccessPeer(&canAccessPeer, peerDevice, currentDevice));
        assert(canAccessPeer);

        HIPCHECK(hipSetDevice(peerDevice));
            HIPCHECK(hipDeviceEnablePeerAccess(currentDevice, 0));
        }

        size_t Nbytes = N*sizeof(char);

        char *A_d0, *A_d1;
        char *A_h;

        A_h = (char*)malloc(Nbytes);

        // allocate and initialize memory on device0
        HIPCHECK (hipSetDevice(currentDevice));
        HIPCHECK (hipMalloc(&A_d0, Nbytes) );
        HIPCHECK ( hipMemset(A_d0, memsetval, Nbytes) ); 

        // allocate and initialize memory on peer device
        HIPCHECK (hipSetDevice(peerDevice));
        HIPCHECK (hipMalloc(&A_d1, Nbytes) );
        HIPCHECK ( hipMemset(A_d1, 0x13, Nbytes) ); 



        // Device0 push to device1, using P2P:
        HIPCHECK (hipSetDevice(p_memcpyWithPeer ? peerDevice : currentDevice));
        HIPCHECK (hipMemcpy(A_d1, A_d0, Nbytes, hipMemcpyDefault));

        // Copy data back to host:
        HIPCHECK (hipSetDevice(peerDevice));
        HIPCHECK (hipMemcpy(A_h, A_d1, Nbytes, hipMemcpyDeviceToHost));

        // Check host data:
        for (int i=0; i<N; i++) {
            if (A_h[i] != memsetval) {
                failed("mismatch at index:%d computed:0x%02x, golden memsetval:0x%02x\n", i, (int)A_h[i], (int)memsetval);
            }
        }
    }


    //---
    // Test which allocated memory first, then enables peer2peer.
    // Enabling peer needs to scan all allocated memory and enable peer access.
    void allocMemoryFirst()
    {
        printf ("\n==testing: %s\n", __func__);
        int deviceCnt;

        HIPCHECK(hipGetDeviceCount(&deviceCnt));

        int currentDevice = p_gpuDevice;
        int peerDevice    = (p_peerDevice == -1) ? ((currentDevice + 1) % deviceCnt) : p_peerDevice;

        printf ("N=%zu  device=%d peerDevice=%d (%d devices total)\n", N, currentDevice, peerDevice, deviceCnt);

        // Must be on a multi-gpu system:
        assert (currentDevice != peerDevice);

        int canAccessPeer;
        HIPCHECK(hipDeviceCanAccessPeer(&canAccessPeer, currentDevice, peerDevice));
        printf ("dev#%d canAccessPeer:#%d=%d\n", currentDevice, peerDevice, canAccessPeer);

        assert(canAccessPeer);

        HIPCHECK (hipSetDevice(currentDevice));
    HIPCHECK(hipDeviceReset());
    HIPCHECK (hipSetDevice(peerDevice));
    HIPCHECK(hipDeviceReset());


    size_t Nbytes = N*sizeof(char);

    char *A_d0, *A_d1;
    char *A_h;

    A_h = (char*)malloc(Nbytes);

    //---
    // allocate and initialize memory on device0
    HIPCHECK (hipSetDevice(currentDevice));
    HIPCHECK (hipMalloc(&A_d0, Nbytes) );
    HIPCHECK ( hipMemset(A_d0, memsetval, Nbytes) ); 

    // allocate and initialize memory on peer device
    HIPCHECK (hipSetDevice(peerDevice));
    HIPCHECK (hipMalloc(&A_d1, Nbytes) );
    HIPCHECK ( hipMemset(A_d1, 0x13, Nbytes) ); 


    //---
    //Enable peer access, for memory already allocated:
    HIPCHECK(hipSetDevice(currentDevice));
    HIPCHECK(hipDeviceEnablePeerAccess(peerDevice, 0));

    if (p_mirrorPeers) {
        int canAccessPeer;
        HIPCHECK(hipDeviceCanAccessPeer(&canAccessPeer, peerDevice, currentDevice));
        assert(canAccessPeer);

        HIPCHECK(hipSetDevice(peerDevice));
        HIPCHECK(hipDeviceEnablePeerAccess(currentDevice, 0));
    }


    //---
    // Copies to test functionality:
    // Device0 push to device1, using P2P:
    HIPCHECK (hipSetDevice(p_memcpyWithPeer ? peerDevice : currentDevice));
    HIPCHECK (hipMemcpy(A_d1, A_d0, Nbytes, hipMemcpyDefault));

    // Copy data back to host:
    HIPCHECK (hipSetDevice(peerDevice));
    HIPCHECK (hipMemcpy(A_h, A_d1, Nbytes, hipMemcpyDeviceToHost));


    //---
    // Check host data:
    for (int i=0; i<N; i++) {
        if (A_h[i] != memsetval) {
            failed("mismatch at index:%d computed:0x%02x, golden memsetval:0x%02x\n", i, (int)A_h[i], (int)memsetval);
        }
    }
}



int main(int argc, char *argv[])
{
    parseMyArguments(argc, argv);

    if (p_tests & 0x1) {
        enablePeerFirst();
    }

    if (p_tests & 0x2) {
        allocMemoryFirst();
    }

    passed();
}
