#pragma once

#include <hc_am.hpp>

typedef int am_status_t;
#define AM_SUCCESS                           0
// TODO - provide better mapping of HSA error conditions to HC error codes.
#define AM_ERROR_MISC                       -1 /** Misellaneous error */

// Flags for am_alloc API:
#define amHostPinned 0x1


namespace hc {

// This is the data that is maintained for each pointer:
struct AmPointerInfo {
    bool        _isDeviceMem;
    void *      _hostPointer;
    void *      _devicePointer;
    size_t      _sizeBytes;
    hc::accelerator _acc;
    unsigned    _allocationFlags;

    AmPointerInfo() {};

    AmPointerInfo(bool isDeviceMem, void *hostPointer, void *devicePointer, size_t sizeBytes, hc::accelerator acc, unsigned allocationFlags) :
        _isDeviceMem(isDeviceMem),
        _hostPointer(hostPointer),
        _devicePointer(devicePointer),
        _sizeBytes(sizeBytes),
        _acc(acc),
        _allocationFlags(allocationFlags)  {};
};
}



namespace hc {


/**
 * Allocates a block of @p size bytes of memory on the specified @p acc.
 *
 * The contents of the newly allocated block of memory are not initialized.
 *
 * If @p size == 0, 0 is returned.
 *
 * Flags must be 0.
 *
 * @returns : On success, pointer to the newly allocated memory is returned.
 * The pointer is typecast to the desired return type.
 *
 * If an error occurred trying to allocate the requested memory, 0 is returned.
 *
 * @see am_free, am_copy
 */
auto_voidp AM_alloc(size_t size, hc::accelerator acc, unsigned flags);

/**
 * Frees a block of memory previously allocated with am_alloc.
 *
 * @see am_alloc, am_copy
 */
am_status_t AM_free(void*  ptr);


/**
 * Copies @p size bytes of memory from @p src to @ dst.  The memory areas (src+size and dst+size) must not overlap.
 *
 * @returns AM_SUCCESS on error or AM_ERROR_MISC if an error occurs.
 * @see am_alloc, am_free
 */
am_status_t AM_copy(void*  dst, const void*  src, size_t size);

am_status_t AM_get_pointer_info(hc::AmPointerInfo *info, void *ptr);


// TODO-implement these:
//am_status_t AM_track_pointer(void* ptr, size_t size, bool isDeviceMem=false, unsigned allocationFlags=0x0);
//am_status_t AM_untrack_pointer(void* ptr);

/**
 * Prints the contents of the memory tracker table to stderr
 *
 * Intended primarily for debug purposes.
 **/
void AM_print_tracker();


}; // namespace hc

