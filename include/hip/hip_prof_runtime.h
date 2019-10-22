/*
Copyright (c) 2019 - present Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef HIP_INCLUDE_HIP_HIP_PROF_RUNTIME_H
#define HIP_INCLUDE_HIP_HIP_PROF_RUNTIME_H

// VDI Op IDs enumeration
enum VdiOpId {
    VDI_OP_ID_DISPATCH = 0,
    VDI_OP_ID_COPY = 1,
    VDI_OP_ID_BARRIER = 2,
    VDI_OP_ID_NUMBER = 3
};

// Types of VDI commands
enum VdiCommandKind {
    VdiCommandKernel = 0x11F0,
    VdiMemcpyDeviceToHost = 0x11F3,
    VdiMemcpyHostToDevice = 0x11F4,
    VdiMemcpyDeviceToDevice = 0x11F5,
    VidMemcpyDeviceToHostRect = 0x1201,
    VdiMemcpyHostToDeviceRect = 0x1202,
    VdiMemcpyDeviceToDeviceRect = 0x1203,
    VdiFillMemory = 0x1207,
}; 

// Activity profiling primitives
typedef void (*InitActivityCallback)(void* id_callback, void* op_callback, void* arg);
typedef bool (*EnableActivityCallback)(uint32_t op, bool enable);
typedef const char* (*GetCmdName)(uint32_t id);

#endif // HIP_INCLUDE_HIP_HIP_PROF_RUNTIME_H

