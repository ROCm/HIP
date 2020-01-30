//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//

#ifndef CL_COMMON_HPP_
#define CL_COMMON_HPP_

#include "top.hpp"
#include "platform/runtime.hpp"
#include "platform/command.hpp"
#include "platform/memory.hpp"
#include "thread/thread.hpp"
#include "platform/commandqueue.hpp"

#include <vector>
#include <utility>

//! \cond ignore
namespace amd {

template <typename T>
class NotNullWrapper
{
private:
    T* const ptrOrNull_;

protected:
    explicit NotNullWrapper(T* ptrOrNull)
        : ptrOrNull_(ptrOrNull)
    { }

public:
    void operator = (T value) const
    {
        if (ptrOrNull_ != NULL) {
            *ptrOrNull_ = value;
        }
    }
};

template <typename T>
class NotNullReference : protected NotNullWrapper<T>
{
public:
    explicit NotNullReference(T* ptrOrNull)
        : NotNullWrapper<T>(ptrOrNull)
    { }

    const NotNullWrapper<T>& operator * () const { return *this; }
};

} // namespace amd

template <typename T>
inline amd::NotNullReference<T>
not_null(T* ptrOrNull)
{
    return amd::NotNullReference<T>(ptrOrNull);
}

#define CL_CHECK_THREAD(thread)                                              \
    (thread != NULL || ((thread = new amd::HostThread()) != NULL             \
            && thread == amd::Thread::current()))

#define RUNTIME_ENTRY_RET(ret, func, args)                                   \
CL_API_ENTRY ret CL_API_CALL                                                 \
func args                                                                    \
{                                                                            \
    amd::Thread* thread = amd::Thread::current();                            \
    if (!CL_CHECK_THREAD(thread)) {                                          \
        *not_null(errcode_ret) = CL_OUT_OF_HOST_MEMORY;                      \
        return (ret) 0;                                                      \
    }

#define RUNTIME_ENTRY_RET_NOERRCODE(ret, func, args)                         \
CL_API_ENTRY ret CL_API_CALL                                                 \
func args                                                                    \
{                                                                            \
    amd::Thread* thread = amd::Thread::current();                            \
    if (!CL_CHECK_THREAD(thread)) {                                          \
        return (ret) 0;                                                      \
    }

#define RUNTIME_ENTRY(ret, func, args)                                       \
CL_API_ENTRY ret CL_API_CALL                                                 \
func args                                                                    \
{                                                                            \
    amd::Thread* thread = amd::Thread::current();                            \
    if (!CL_CHECK_THREAD(thread)) {                                          \
        return CL_OUT_OF_HOST_MEMORY;                                        \
    }

#define RUNTIME_ENTRY_VOID(ret, func, args)                                  \
CL_API_ENTRY ret CL_API_CALL                                                 \
func args                                                                    \
{                                                                            \
    amd::Thread* thread = amd::Thread::current();                            \
    if (!CL_CHECK_THREAD(thread)) {                                          \
        return;                                                              \
    }

#define RUNTIME_EXIT                                                         \
    /* FIXME_lmoriche: we should check to thread->lastError here! */         \
}

//! Helper function to check "properties" parameter in various functions
int checkContextProperties(
    const cl_context_properties *properties,
    bool*   offlineDevices);

namespace amd {

namespace detail {

template <typename T>
struct ParamInfo
{
    static inline std::pair<const void*, size_t> get(const T& param) {
        return std::pair<const void*, size_t>(&param, sizeof(T));
    }
};

template <>
struct ParamInfo<const char*>
{
    static inline std::pair<const void*, size_t> get(const char* param) {
        return std::pair<const void*, size_t>(param, strlen(param) + 1);
    }
};

template <int N>
struct ParamInfo<char[N]>
{
    static inline std::pair<const void*, size_t> get(const char* param) {
        return std::pair<const void*, size_t>(param, strlen(param) + 1);
    }
};

} // namespace detail

template <typename T>
static inline cl_int
clGetInfo(
    T& field,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    const void *valuePtr;
    size_t valueSize;

    std::tie(valuePtr, valueSize)
        = detail::ParamInfo<typename std::remove_const<T>::type>::get(field);

    *not_null(param_value_size_ret) = valueSize;

    cl_int ret = CL_SUCCESS;
    if (param_value != NULL && param_value_size < valueSize) {
        if (!std::is_pointer<T>() || !std::is_same<typename std::remove_const<
                typename std::remove_pointer<T>::type>::type, char>()) {
            return CL_INVALID_VALUE;
        }
	// For char* and char[] params, we will at least fill up to
        // param_value_size, then return an error.
        valueSize = param_value_size;
        static_cast<char*>(param_value)[--valueSize] = '\0';
        ret = CL_INVALID_VALUE;
    }

    if (param_value != NULL) {
        ::memcpy(param_value, valuePtr, valueSize);
        if (param_value_size > valueSize) {
            ::memset(static_cast<address>(param_value) + valueSize,
                '\0', param_value_size - valueSize);
        }
    }

    return ret;
}

static inline cl_int
clSetEventWaitList(
    Command::EventWaitList& eventWaitList,
    const amd::HostQueue& hostQueue,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list)
{
    if ((num_events_in_wait_list == 0 && event_wait_list != NULL)
            || (num_events_in_wait_list != 0 && event_wait_list == NULL)) {
        return CL_INVALID_EVENT_WAIT_LIST;
    }

    while (num_events_in_wait_list-- > 0) {
        cl_event event = *event_wait_list++;
        Event* amdEvent = as_amd(event);
        if (!is_valid(event)) {
            return CL_INVALID_EVENT_WAIT_LIST;
        }
        if (&hostQueue.context() != &amdEvent->context()) {
            return CL_INVALID_CONTEXT;
        }
        if ((amdEvent->command().queue() != &hostQueue) && !amdEvent->notifyCmdQueue()) {
            return CL_INVALID_EVENT_WAIT_LIST;
        }
        eventWaitList.push_back(amdEvent);
    }
    return CL_SUCCESS;
}

//! Common function declarations for CL-external graphics API interop
cl_int clEnqueueAcquireExtObjectsAMD(cl_command_queue command_queue,
    cl_uint num_objects, const cl_mem* mem_objects,
    cl_uint num_events_in_wait_list, const cl_event* event_wait_list,
    cl_event* event, cl_command_type cmd_type);
cl_int clEnqueueReleaseExtObjectsAMD(cl_command_queue command_queue,
    cl_uint num_objects, const cl_mem* mem_objects,
    cl_uint num_events_in_wait_list, const cl_event* event_wait_list,
    cl_event* event, cl_command_type cmd_type);

// This may need moving somewhere tidier...

struct PlatformIDS { const struct KHRicdVendorDispatchRec* dispatch_; };
class PlatformID {
public:
    static PlatformIDS Platform;
};
#define AMD_PLATFORM (reinterpret_cast<cl_platform_id>(&amd::PlatformID::Platform))

} // namespace amd

extern "C" {

extern CL_API_ENTRY cl_key_amd CL_API_CALL
clCreateKeyAMD(
    cl_platform_id platform,
    void (CL_CALLBACK * destructor)( void * ),
    cl_int * errcode_ret);

extern CL_API_ENTRY cl_int CL_API_CALL
clObjectGetValueForKeyAMD(
    void * object,
    cl_key_amd key,
    void ** ret_val);

extern CL_API_ENTRY cl_int CL_API_CALL
clObjectSetValueForKeyAMD(
    void * object,
    cl_key_amd key,
    void * value);

#if defined(CL_VERSION_1_1)
extern CL_API_ENTRY cl_int CL_API_CALL
clSetCommandQueueProperty(
    cl_command_queue command_queue,
    cl_command_queue_properties properties,
    cl_bool enable,
    cl_command_queue_properties *old_properties) CL_API_SUFFIX__VERSION_1_0;
#endif // CL_VERSION_1_1

extern CL_API_ENTRY cl_mem CL_API_CALL
clConvertImageAMD(
    cl_context              context,
    cl_mem                  image,
    const cl_image_format * image_format,
    cl_int *                errcode_ret);

extern CL_API_ENTRY cl_mem CL_API_CALL
clCreateBufferFromImageAMD(
    cl_context              context,
    cl_mem                  image,
    cl_int *                errcode_ret);

extern CL_API_ENTRY cl_program CL_API_CALL
clCreateProgramWithAssemblyAMD(
    cl_context              context,
    cl_uint                 count,
    const char **           strings,
    const size_t *          lengths,
    cl_int *                errcode_ret);

} // extern "C"

//! \endcond

#endif /*CL_COMMON_HPP_*/
