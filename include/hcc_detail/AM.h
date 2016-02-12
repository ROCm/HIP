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
    void *      _hostPointer;   ///< Host pointer.  If host access is not allowed, NULL.
    void *      _devicePointer; ///< Device pointer.  
    size_t      _sizeBytes;     ///< Size of allocation.
    hc::accelerator _acc;       ///< Device / Accelerator to use.
    bool        _isInDeviceMem;    ///< Memory is physically resident on a device (if false, memory is located on host)
    bool        _isAmManaged;   ///< Memory was allocated by AM and should be freed when am_reset is called.

    int         _appId;              ///< App-specific storage.  Used by HIP to store deviceID.
    unsigned    _appAllocationFlags; ///< App-specific allocation flags.  Used by HIP to store allocation flags. 

    AmPointerInfo() {};

    AmPointerInfo(void *hostPointer, void *devicePointer, size_t sizeBytes, hc::accelerator acc, bool isInDeviceMem, bool isAmManaged) :
        _hostPointer(hostPointer),
        _devicePointer(devicePointer),
        _sizeBytes(sizeBytes),
        _acc(acc),
        _isInDeviceMem(isInDeviceMem),
        _isAmManaged(isAmManaged),
        _appId(-1),
        _appAllocationFlags(0)  {};
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


/**
 * Return information about tracked pointer.
 *
 * AM tracks pointers when they are allocated or added to tracker with am_track_pointer.
 * The tracker tracks the base pointer as well as the size of the allocation, and will
 * find the information for a pointer anywhere in the tracked range.
 *
 * @returns AM_ERROR_MISC if pointer is not currently being tracked.
 * @returns AM_SUCCESS if pointer is tracked and writes info to @p info.
 *
 * @see AM_memtracker_add, 
 */
am_status_t am_memtracker_getinfo(hc::AmPointerInfo *info, void *ptr);


//TODO-doc
am_status_t am_memtracker_add(void* ptr, size_t sizeBytes, hc::accelerator acc, bool isDeviceMem=false);


//TODO-doc
am_status_t am_memtracker_update(void* ptr, int appId, unsigned allocationFlags);


/** 
 * Remove the pointer from the tracker structure.
 *
 * @p ptr may be anywhere in a tracked memory range.
 *
 * @returns AM_ERROR_MISC if pointer is not found in tracker.  
 * @returns AM_SUCCESS if pointer is not found in tracker.  
 */
am_status_t am_memtracker_remove(void* ptr);

/**
 * Remove all memory allocations associated with specified accelerator.
 *
 * @returns Number of entries reset.
 */
size_t am_memtracker_reset(hc::accelerator acc);

/**
 * Prints info about the memory tracker table.
 *
 * Intended primarily for debug purposes.
 **/
void am_memtracker_print();

void am_memtracker_sizeinfo(hc::accelerator acc, size_t *deviceMemSize, size_t *hostMemSize, size_t *userMemSize);


}; // namespace hc

