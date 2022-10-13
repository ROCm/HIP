/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>
#include <kernels.hh>
#include <resource_guards.hh>
#include <utils.hh>

TEST_CASE("Unit_hipStreamAttachMemAsync_Positive_Basic") {
    if (!DeviceAttributesSupport(0, hipDeviceAttributeManagedMemory)) {
        HipTest::HIP_SKIP_TEST("Managed memory is not supported");
        return;
    }

    StreamGuard stream(Streams::created);
    LinearAllocGuard<hipDeviceptr_t> managed(LinearAllocs::hipMallocManaged, kPageSize, hipMemAttachHost);

    HIP_CHECK(hipStreamAttachMemAsync(stream.stream(), managed.ptr(), 0));
    HIP_CHECK(hipStreamSynchronize(stream.stream()));
}

TEST_CASE("Unit_hipStreamAttachMemAsync_Positive_Pageable") {
    if (!DeviceAttributesSupport(0, hipDeviceAttributeManagedMemory)) {
        HipTest::HIP_SKIP_TEST("Managed memory is not supported");
        return;
    }

    if (!DeviceAttributesSupport(0, hipDeviceAttributePageableMemoryAccess)) {
        HipTest::HIP_SKIP_TEST("Pageable memory access is not supported");
        return;
    }

    StreamGuard stream(Streams::created);
    LinearAllocGuard<hipDeviceptr_t> pageable(LinearAllocs::malloc, kPageSize);

    HIP_CHECK(hipStreamAttachMemAsync(stream.stream(), pageable.ptr(), kPageSize));
    HIP_CHECK(hipStreamSynchronize(stream.stream()));
}

// CUDA docs:
// If the cudaMemAttachGlobal flag is specified, the memory can be accessed by any stream on any 
// device.
TEST_CASE("Unit_hipStreamAttachMemAsync_Positive_AttachGlobal") {
    if (!DeviceAttributesSupport(0, hipDeviceAttributeManagedMemory)) {
        HipTest::HIP_SKIP_TEST("Managed memory is not supported");
        return;
    }

    const auto device_count = HipTest::getDeviceCount();
    const auto stream_count = device_count < 2 ? 8 : device_count;

    std::vector<std::unique_ptr<StreamGuard>> streams;
    streams.reserve(stream_count);
    for (int i = 0; i < stream_count; ++i) {
        if (device_count > 1) {
            HIP_CHECK(hipSetDevice(i));
        }
        streams.push_back(std::make_unique<StreamGuard>(Streams::created));
    }
    
    LinearAllocGuard<int> managed_global(LinearAllocs::hipMallocManaged, sizeof(int) * stream_count, hipMemAttachHost);

    HIP_CHECK(hipStreamAttachMemAsync(nullptr, reinterpret_cast<hipDeviceptr_t*>(managed_global.ptr()), 0, hipMemAttachGlobal));
    HIP_CHECK(hipStreamSynchronize(nullptr));

    for (int i = 0; i < stream_count; ++i) {
        HipTest::launchKernel(Set, 1, 1, 0, streams.at(i)->stream(), managed_global.ptr() + i, i);
    }

    for (auto&& stream : streams) {
        HIP_CHECK(hipStreamSynchronize(stream->stream()));
    }

    for (int i = 0; i < stream_count; ++i) {
        REQUIRE(managed_global.ptr()[i] == i);
    }
}

// CUDA docs:
// If the cudaMemAttachHost flag is specified, the program makes a guarantee that it won't access 
// the memory on the device from any stream on a device that has a zero value for the device 
// attribute cudaDevAttrConcurrentManagedAccess.
TEST_CASE("Unit_hipStreamAttachMemAsync_Positive_AttachHost") {
    if (!DeviceAttributesSupport(0, hipDeviceAttributeManagedMemory)) {
        HipTest::HIP_SKIP_TEST("Managed memory is not supported");
        return;
    }

    if (DeviceAttributesSupport(0, hipDeviceAttributeConcurrentManagedAccess)) {
        HipTest::HIP_SKIP_TEST("Device supports concurrent managed access");
        return;
    }

    StreamGuard stream(Streams::created);
    LinearAllocGuard<int> managed_global(LinearAllocs::hipMallocManaged, sizeof(int));
    LinearAllocGuard<int> managed_host(LinearAllocs::hipMallocManaged, sizeof(int));

    HIP_CHECK(hipStreamAttachMemAsync(stream.stream(), reinterpret_cast<hipDeviceptr_t*>(managed_host.ptr()), 0, hipMemAttachHost));
    HIP_CHECK(hipStreamSynchronize(stream.stream()));

    HipTest::launchKernel(Set, 1, 1, 0, stream.stream(), managed_global.ptr(), 32);
    *managed_host.ptr() = 64;
    HIP_CHECK(hipStreamSynchronize(stream.stream()));
    
    REQUIRE(*managed_global.ptr() == 32);
    REQUIRE(*managed_host.ptr() == 64);
}

// CUDA docs:
// If the cudaMemAttachSingle flag is specified and stream is associated with a device that has a 
// zero value for the device attribute cudaDevAttrConcurrentManagedAccess, the program makes a 
// guarantee that it will only access the memory on the device from stream.
TEST_CASE("Unit_hipStreamAttachMemAsync_Positive_AttachSingle") {
    if (!DeviceAttributesSupport(0, hipDeviceAttributeManagedMemory)) {
        HipTest::HIP_SKIP_TEST("Managed memory is not supported");
        return;
    }

    if (DeviceAttributesSupport(0, hipDeviceAttributeConcurrentManagedAccess)) {
        HipTest::HIP_SKIP_TEST("Device supports concurrent managed access");
        return;
    }
    
    StreamGuard stream1(Streams::created);
    StreamGuard stream2(Streams::created);

    LinearAllocGuard<int> managed_global(LinearAllocs::hipMallocManaged, sizeof(int));
    LinearAllocGuard<int> managed_single(LinearAllocs::hipMallocManaged, sizeof(int), hipMemAttachHost);

    HIP_CHECK(hipStreamAttachMemAsync(stream1.stream(), reinterpret_cast<hipDeviceptr_t*>(managed_single.ptr()), 0, hipMemAttachSingle));
    HIP_CHECK(hipStreamSynchronize(stream1.stream()));

    HipTest::launchKernel(Set, 1, 1, 0, stream1.stream(), managed_single.ptr(), 64);
    HIP_CHECK(hipStreamSynchronize(stream1.stream()));
    
    HipTest::launchKernel(Set, 1, 1, 0, stream2.stream(), managed_global.ptr(), 32);
    
    REQUIRE(*managed_single.ptr() == 64);
    *managed_single.ptr() = 128;

    HIP_CHECK(hipStreamSynchronize(stream2.stream()));
    
    REQUIRE(*managed_global.ptr() == 32);
    REQUIRE(*managed_single.ptr() == 128);
}

TEST_CASE("Unit_hipStreamAttachMemAsync_Negative_Parameters") {
    if (!DeviceAttributesSupport(0, hipDeviceAttributeManagedMemory)) {
        HipTest::HIP_SKIP_TEST("Managed memory is not supported");
        return;
    }

    StreamGuard stream(Streams::created);
    LinearAllocGuard<hipDeviceptr_t> managed(LinearAllocs::hipMallocManaged, kPageSize, hipMemAttachHost);

    SECTION("invalid stream") {
        HIP_CHECK(hipStreamDestroy(stream.stream()));
        HIP_CHECK_ERROR(hipStreamAttachMemAsync(stream.stream(), managed.ptr()), hipErrorContextIsDestroyed);
    }

    SECTION("dev_ptr == nullptr") {
        HIP_CHECK_ERROR(hipStreamAttachMemAsync(stream.stream(), nullptr), hipErrorInvalidValue);
    }

    SECTION("length is not zero nor entire allocation size") {
        HIP_CHECK_ERROR(hipStreamAttachMemAsync(stream.stream(), managed.ptr(), kPageSize / 2), hipErrorInvalidValue);
    }

    SECTION("invalid flags") {
        HIP_CHECK_ERROR(hipStreamAttachMemAsync(stream.stream(), managed.ptr(), 0, hipMemAttachGlobal | hipMemAttachHost | hipMemAttachSingle), hipErrorInvalidValue);
    }

    SECTION("attach single to nullstream") {
        HIP_CHECK_ERROR(hipStreamAttachMemAsync(nullptr, managed.ptr(), 0, hipMemAttachSingle), hipErrorInvalidValue);
    }

    LinearAllocGuard<hipDeviceptr_t> pageable(LinearAllocs::malloc, kPageSize);
    
    if (!DeviceAttributesSupport(0, hipDeviceAttributePageableMemoryAccess)) {
        SECTION("dev_ptr is pageable memory") {
            HIP_CHECK_ERROR(hipStreamAttachMemAsync(stream.stream(), pageable.ptr(), kPageSize), hipErrorInvalidValue);
        }
    } else {
        SECTION("length is zero for pageable memory") {
            HIP_CHECK_ERROR(hipStreamAttachMemAsync(stream.stream(), pageable.ptr(), 0), hipErrorInvalidValue);
        }
    }
}