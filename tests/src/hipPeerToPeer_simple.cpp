/*
Copyright (c) 2015-2017 Advanced Micro Devices, Inc. All rights reserved.

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

/* HIT_START
 * BUILD: %t %s test_common.cpp
 * RUN: %t EXCLUDE_HIP_PLATFORM all
 * RUN: %t --memcpyWithPeer EXCLUDE_HIP_PLATFORM all
 * RUN: %t --mirrorPeers EXCLUDE_HIP_PLATFORM all
 * HIT_END
 */

#include "hip/hip_runtime.h"
#include "test_common.h"

bool p_memcpyWithPeer = false; // use the peer device for the P2P copy
bool p_mirrorPeers = false; // in addition to mapping current to peer space, map peer to current space.
int  p_peerDevice = -1;  // explicly specify which peer to use, else use p_gpuDevice + 1.


int g_currentDevice;
int g_peerDevice;

void help(char *argv[])
{
    printf ("usage: %s [OPTIONS]\n", argv[0]);
    printf (" --memcpyWithPeer : Perform memcpy with peer.\n");
    printf (" --mirrorPeers    : Mirror memory onto both default device and peerdevice.  If 0, memory is mapped only on the default device.\n");
    printf (" --peerDevice N   : Set peer device.\n");
};


static hipError_t myHipMemcpy(void *dest, const void *src, size_t sizeBytes, hipMemcpyKind kind,  hipStream_t stream, bool async)
{
    if (async) {
		hipError_t e = hipMemcpyAsync(dest, src, sizeBytes, kind, stream);
        //HIPCHECK(hipStreamSynchronize(stream));
        return (e);
    } else {
		return hipMemcpy(dest, src, sizeBytes, kind);
    };
}


void parseMyArguments(int argc, char *argv[])
{
    int more_argc = HipTest::parseStandardArguments(argc, argv, false);
    // parse args for this test:
    for (int i = 1; i < more_argc; i++) {
        const char *arg = argv[i];

        if (!strcmp(arg, "--help")) {
            help(argv);
            exit(-1);
        } else if (!strcmp(arg, "--memcpyWithPeer")) {
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

void syncBothDevices()
{
    int saveDevice;
    HIPCHECK(hipGetDevice(&saveDevice));
    HIPCHECK(hipSetDevice(g_currentDevice));
    HIPCHECK(hipDeviceSynchronize());

    HIPCHECK(hipSetDevice(g_peerDevice));
    HIPCHECK(hipDeviceSynchronize());

    HIPCHECK(hipSetDevice(saveDevice));
}


// Sets globals g_currentDevice, g_peerDevice
void setupPeerTests()
{
    int deviceCnt;

    HIPCHECK(hipGetDeviceCount(&deviceCnt));

    g_currentDevice = p_gpuDevice;
    g_peerDevice    = (p_peerDevice == -1) ? ((g_currentDevice + 1) % deviceCnt) : p_peerDevice;

    printf ("N=%zu  device=%d peerDevice=%d (%d devices total)\n", N, g_currentDevice, g_peerDevice, deviceCnt);

    // Must be on a multi-gpu system:
    assert (g_currentDevice != g_peerDevice);

    int canAccessPeer;
    HIPCHECK(hipDeviceCanAccessPeer(&canAccessPeer, g_currentDevice, g_peerDevice));
    printf ("dev#%d canAccessPeer:#%d=%d\n", g_currentDevice, g_peerDevice, canAccessPeer);

    assert(canAccessPeer);

    HIPCHECK (hipSetDevice(g_currentDevice));
    HIPCHECK(hipDeviceReset());
    HIPCHECK (hipSetDevice(g_peerDevice));
    HIPCHECK(hipDeviceReset());
}

//---
// Test which enables peer2peer first, then allocates the memory.
void enablePeerFirst(bool useAsyncCopy)
{
    printf ("\n==testing: %s useAsyncCopy=%d\n", __func__, useAsyncCopy);

    setupPeerTests();

    // Always enable g_currentDevice to see the allocations on peerDevice.
    HIPCHECK(hipSetDevice(g_currentDevice));
    HIPCHECK(hipDeviceEnablePeerAccess(g_peerDevice, 0));

    if (p_mirrorPeers) {
        // Mirror peers allows the peer device to see the allocations on currentDevice.
        int canAccessPeer;
        HIPCHECK(hipDeviceCanAccessPeer(&canAccessPeer, g_peerDevice, g_currentDevice));
        assert(canAccessPeer);

        HIPCHECK(hipSetDevice(g_peerDevice));
        HIPCHECK(hipDeviceEnablePeerAccess(g_currentDevice, 0));
    }

    size_t Nbytes = N*sizeof(char);

    char *A_d0, *A_d1;
    char *A_h;

    A_h = (char*)malloc(Nbytes);

    // allocate and initialize memory on device0
    HIPCHECK (hipSetDevice(g_currentDevice));
    HIPCHECK (hipMalloc(&A_d0, Nbytes) );
    HIPCHECK (hipMemset(A_d0, memsetval, Nbytes) );

    // allocate and initialize memory on peer device
    HIPCHECK (hipSetDevice(g_peerDevice));
    HIPCHECK (hipMalloc(&A_d1, Nbytes) );
    HIPCHECK (hipMemset(A_d1, 0x13, Nbytes) );



    // Device0 push to device1, using P2P:
    // NOTE : if p_mirrorPeers=0 and p_memcpyWithPeer=1, then peer device does not have mapping for A_d1 and we need to use a
    //        a host staging copy for the P2P access.
    HIPCHECK (hipSetDevice(p_memcpyWithPeer ? g_peerDevice : g_currentDevice));
    HIPCHECK (myHipMemcpy(A_d1, A_d0, Nbytes, hipMemcpyDefault, 0/*stream*/, useAsyncCopy)); // This is P2P copy.

    // Copy data back to host:
    // Have to wait for previous operation to finish, since we are switching to another one:
    HIPCHECK(hipDeviceSynchronize());

    HIPCHECK (hipSetDevice(g_peerDevice));
    HIPCHECK (myHipMemcpy(A_h, A_d1, Nbytes, hipMemcpyDeviceToHost, 0/*stream*/, useAsyncCopy));
    HIPCHECK(hipDeviceSynchronize());

    HIPCHECK (hipSetDevice(g_currentDevice));

    // Check host data:
    for (int i=0; i<N; i++) {
        if (A_h[i] != memsetval) {
            failed("mismatch at index:%d computed:0x%02x, golden memsetval:0x%02x\n", i, (int)A_h[i], (int)memsetval);
        }
    }

    printf ("==done: %s useAsyncCopy:%d\n\n", __func__, useAsyncCopy);
}


//---
// Test which allocated memory first, then enables peer2peer.
// Enabling peer needs to scan all allocated memory and enable peer access.
void allocMemoryFirst(bool useAsyncCopy)
{
    printf ("\n==testing: %s useAsyncCopy=%d\n", __func__, useAsyncCopy);

    setupPeerTests();

    size_t Nbytes = N*sizeof(char);

    char *A_d0, *A_d1;
    char *A_h;

    A_h = (char*)malloc(Nbytes);

    //---
    // allocate and initialize memory on device0
    HIPCHECK (hipSetDevice(g_currentDevice));
    HIPCHECK (hipMalloc(&A_d0, Nbytes) );
    HIPCHECK ( hipMemset(A_d0, memsetval, Nbytes) );

    // allocate and initialize memory on peer device
    HIPCHECK (hipSetDevice(g_peerDevice));
    HIPCHECK (hipMalloc(&A_d1, Nbytes) );
    HIPCHECK ( hipMemset(A_d1, 0x13, Nbytes) );


    //---
    //Enable peer access, for memory already allocated:
    HIPCHECK(hipSetDevice(g_currentDevice));
    HIPCHECK(hipDeviceEnablePeerAccess(g_peerDevice, 0));

    if (p_mirrorPeers) {
        int canAccessPeer;
        HIPCHECK(hipDeviceCanAccessPeer(&canAccessPeer, g_peerDevice, g_currentDevice));
        assert(canAccessPeer);

        HIPCHECK(hipSetDevice(g_peerDevice));
        HIPCHECK(hipDeviceEnablePeerAccess(g_currentDevice, 0));
    }


    //---
    // Copies to test functionality:
    // Device0 push to device1, using P2P:
    HIPCHECK (hipSetDevice(p_memcpyWithPeer ? g_peerDevice : g_currentDevice));
    HIPCHECK (myHipMemcpy(A_d1, A_d0, Nbytes, hipMemcpyDefault, 0/*stream*/, useAsyncCopy));

    syncBothDevices(); // TODO - remove me, should handle this in implementation.

    // Copy data back to host:
    HIPCHECK (hipSetDevice(g_peerDevice));
    HIPCHECK (myHipMemcpy(A_h, A_d1, Nbytes, hipMemcpyDeviceToHost, 0/*stream*/, useAsyncCopy));

    syncBothDevices(); // TODO - remove me, should handle this in implementation.


    //---
    // Check host data:
    for (int i=0; i<N; i++) {
        if (A_h[i] != memsetval) {
            failed("mismatch at index:%d computed:0x%02x, golden memsetval:0x%02x\n", i, (int)A_h[i], (int)memsetval);
        }
    }
    printf ("==done: %s useAsyncCopy=%d\n\n", __func__, useAsyncCopy);
}


//---
// Test which tests peer H2D copy - ie: copy-engine=1, dst=1, src=0 (Host)
// A_d0 is pinned host on dev0 (this)
// A_d1 is device memory on dev1 (peer)
//
void testPeerHostToDevice(bool useAsyncCopy)
{
    printf ("\n==testing: %s useAsyncCopy=%d\n", __func__, useAsyncCopy);

    setupPeerTests();

    // Always enable g_currentDevice to see the allocations on peerDevice.
    HIPCHECK(hipSetDevice(g_currentDevice));
    HIPCHECK(hipDeviceEnablePeerAccess(g_peerDevice, 0));

    if (p_mirrorPeers) {
        // Mirror peers allows the peer device to see the allocations on currentDevice.
        int canAccessPeer;
        HIPCHECK(hipDeviceCanAccessPeer(&canAccessPeer, g_peerDevice, g_currentDevice));
        assert(canAccessPeer);

        HIPCHECK(hipSetDevice(g_peerDevice));
        HIPCHECK(hipDeviceEnablePeerAccess(g_currentDevice, 0));
    }

    size_t Nbytes = N*sizeof(char);

    char *A_host_d0, *A_d1;
    char *A_h;

    A_h = (char*)malloc(Nbytes);

    // allocate and initialize memory on device0
    HIPCHECK (hipSetDevice(g_currentDevice));
    HIPCHECK (hipHostMalloc(&A_host_d0, Nbytes) );
    HIPCHECK (hipMemset(A_host_d0, memsetval, Nbytes) );

    // allocate and initialize memory on peer device
    HIPCHECK (hipSetDevice(g_peerDevice));
    HIPCHECK (hipMalloc(&A_d1, Nbytes) );
    HIPCHECK (hipMemset(A_d1, 0x13, Nbytes) );

    bool firstAsyncCopy = useAsyncCopy; /*TODO - should be useAsyncCopy*/

    syncBothDevices();



    // Device0 push to device1, using P2P:
    // NOTE : if p_mirrorPeers=0 and p_memcpyWithPeer=1, then peer device does not have mapping for A_d1 and we need to use a
    //        a host staging copy for the P2P access.
    if (p_memcpyWithPeer) {
        // p_memcpyWithPeer=1 case is HostToDevice.
        // if p_mirrorPeers = 1, this is accelerated copy over PCIe.
        // if p_mirrorPeers = 0, this should fall back to host (because peer can't see A_host_d0)
        HIPCHECK (hipSetDevice(g_peerDevice));
        HIPCHECK (myHipMemcpy(A_d1, A_host_d0, Nbytes, hipMemcpyHostToDevice, 0/*stream*/, firstAsyncCopy)); // This is P2P copy.
    } else {
        // p_memcpyWithPeer=0 case is HostToDevice.
        // if p_mirrorPeers = 1, this is accelerated copy over PCIe.
        // if p_mirrorPeers = 0, this should fall back to host (because device0 can't see A_d1)
        HIPCHECK (hipSetDevice(g_currentDevice));
        HIPCHECK (myHipMemcpy(A_d1, A_host_d0, Nbytes, hipMemcpyHostToDevice, 0/*stream*/, firstAsyncCopy)); // This is P2P copy.
    }

    syncBothDevices();

    // Copy data back to host:
    HIPCHECK (hipSetDevice(g_peerDevice));
    HIPCHECK (myHipMemcpy(A_h, A_d1, Nbytes, hipMemcpyDeviceToHost, 0/*stream*/, useAsyncCopy));
    HIPCHECK(hipDeviceSynchronize());

    HIPCHECK (hipSetDevice(g_currentDevice));
    HIPCHECK(hipDeviceSynchronize());

    // Check host data:
    for (int i=0; i<N; i++) {
        if (A_h[i] != memsetval) {
            failed("mismatch at index:%d computed:0x%02x, golden memsetval:0x%02x\n", i, (int)A_h[i], (int)memsetval);
        }
    }

    printf ("==done: %s useAsyncCopy:%d\n\n", __func__, useAsyncCopy);
}



void simpleNegative()
{
    printf ("\n==testing: %s\n", __func__);

    setupPeerTests();

    int deviceId;
    HIPCHECK (hipGetDevice(&deviceId));

    //---
    //-- self is not a peer
    int canAccessPeer;
    hipError_t e = hipDeviceCanAccessPeer(&canAccessPeer, deviceId, deviceId);
    HIPASSERT( e == hipSuccess);  // no error returned, it doesn't hurt to ask.
    HIPASSERT (canAccessPeer == 0); // but self is not a peer.

    e = hipSuccess;
    //---
    // Enable same device twice in a row:
    HIPCHECK(hipSetDevice(g_currentDevice));
    HIPCHECK(hipDeviceEnablePeerAccess(g_peerDevice, 0));
    e =(hipDeviceEnablePeerAccess(g_peerDevice, 0));
    HIPASSERT (e == hipErrorPeerAccessAlreadyEnabled);

    //---
    // try disabling twice in a row
    HIPCHECK(hipDeviceDisablePeerAccess(g_peerDevice));
    e =(hipDeviceDisablePeerAccess(g_peerDevice));
    HIPASSERT (e == hipErrorPeerAccessNotEnabled);


    // More tests here:
    printf ("==done: %s\n\n", __func__);
}



int main(int argc, char *argv[])
{
    parseMyArguments(argc, argv);


    if (p_tests & 0x100) {
        testPeerHostToDevice(false/*useAsyncCopy*/);
    }
    testPeerHostToDevice(true/*useAsyncCopy*/);

    if (p_tests & 0x1) {
        enablePeerFirst(false/*useAsyncCopy*/);
    }

    if (p_tests & 0x2) {
        allocMemoryFirst(false/*useAsyncCopy*/);
    }

    if (p_tests & 0x4) {
        simpleNegative();
    }

    if (p_tests & 0x8) {
        enablePeerFirst(true/*useAsyncCopy*/);
    }
    if (p_tests & 0x10) {
        allocMemoryFirst(true/*useAsyncCopy*/);
    }

    passed();
}
