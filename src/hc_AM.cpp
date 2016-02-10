
#include "hc_am.hpp"
#include "hsa.h"


#include "hcc_detail/AM.h" // TODO - Remove me.

#define DB_TRACKER 1

#if DB_TRACKER 
#define mprintf( ...) {\
        fprintf (stderr, __VA_ARGS__);\
        };
#else
#define mprintf( ...) 
#endif

//=========================================================================================================
// Pointer Tracker Structures:
//=========================================================================================================
#include <map>
#include <iostream>
//#include <shared_mutex>

struct AmMemoryRange {
    void * _basePointer;
    void * _endPointer;
    AmMemoryRange(void *basePointer, size_t sizeBytes) :
        _basePointer(basePointer), _endPointer((unsigned char*)basePointer + sizeBytes - 1) {};
};

// Functor to compare ranges:
struct AmMemoryRangeCompare {
    // Return true is LHS range is less than RHS - used to order the 
    bool operator()(const AmMemoryRange &lhs, const AmMemoryRange &rhs) const
    {
        return lhs._endPointer < rhs._basePointer;
    }

};


std::ostream &operator<<(std::ostream &os, const hc::AmPointerInfo &ap)
{
    os << "hostPointer:" << ap._hostPointer << " devicePointer:"<< ap._devicePointer << " sizeBytes:" << ap._sizeBytes
       << " isDeviceMem:" << ap._isDeviceMem  << " allocFlags:" << ap._allocationFlags;
    return os;
}



// This structure tracks information for each pointer.
// Uses memory-range-based lookups - so pointers that exist anywhere in the range of hostPtr + size will find the associated AmPointerInfo.
// The insertions and lookups use a self-balancing binary tree and should support O(logN) lookup speed.
// The structure is thread-safe - writers obtain a mutex before modifying the tree.  Multiple simulatenous readers are supported.
class AmPointerTracker {
typedef std::map<AmMemoryRange, hc::AmPointerInfo, AmMemoryRangeCompare> MapTrackerType;
public:

    void insert(void *pointer, const hc::AmPointerInfo &p);
    int remove(void *pointer);

    MapTrackerType::iterator find(void *hostPtr);

    MapTrackerType::iterator end() { return _tracker.end(); };

    std::ostream & print (std::ostream &os);
private:
    MapTrackerType  _tracker;
    //std::shared_timed_mutex _mut;
};


//---
void AmPointerTracker::insert (void *pointer, const hc::AmPointerInfo &p)
{
    // TODO-mutex - write lock.
    mprintf ("insert: %p + %zu\n", pointer, p._sizeBytes);
    _tracker.insert(std::make_pair(AmMemoryRange(pointer, p._sizeBytes), p));


}


//---
// Return 1 if removed or 0 if not found.
int AmPointerTracker::remove (void *pointer)
{
    // TODO-mutex - write lock.
    mprintf ("remove: %p\n", pointer);
    return _tracker.erase(AmMemoryRange(pointer,1));
}


//---
AmPointerTracker::MapTrackerType::iterator  AmPointerTracker::find (void *pointer)
{
    // TODO-mutex- read lock
    auto iter = _tracker.find(AmMemoryRange(pointer,1));
    mprintf ("find: %p\n", pointer);
    return iter;
}


std::ostream & AmPointerTracker::print (std::ostream &os)
{
    for (auto iter = _tracker.begin() ; iter != _tracker.end(); iter++) {
        os << "  " << iter->first._basePointer << "..." << iter->first._endPointer << "::  ";
        os << iter->second << std::endl;
    }

    return os;
}


//=========================================================================================================
// Global var defs:
//=========================================================================================================
AmPointerTracker g_amPointerTracker;  // Track all am pointer allocations.


//=========================================================================================================
// API Definitions.
//=========================================================================================================
//
//

namespace hc {

// Allocate accelerator memory, return NULL if memory could not be allocated:
auto_voidp AM_alloc(size_t sizeBytes, hc::accelerator acc, unsigned flags) 
{

    void *ptr = NULL;

    if (sizeBytes != 0 ) {
        if (acc.is_hsa_accelerator()) {
            hsa_agent_t *hsa_agent = static_cast<hsa_agent_t*> (acc.get_default_view().get_hsa_agent());
            hsa_region_t *alloc_region;
            if (flags & amHostPinned) {
               alloc_region = static_cast<hsa_region_t*>(acc.get_hsa_am_system_region());
            } else {
               alloc_region = static_cast<hsa_region_t*>(acc.get_hsa_am_region());
            }

            if (alloc_region->handle != -1) {

                hsa_status_t s1 = hsa_memory_allocate(*alloc_region, sizeBytes, &ptr);
                hsa_status_t s2 = hsa_memory_assign_agent(ptr, *hsa_agent, HSA_ACCESS_PERMISSION_RW);

                if ((s1 != HSA_STATUS_SUCCESS) || (s2 != HSA_STATUS_SUCCESS)) {
                    ptr = NULL;
                } else {
                    if (flags & amHostPinned) {
                        g_amPointerTracker.insert(ptr, 
                                hc::AmPointerInfo(false/*isDevice*/, ptr/*hostPointer*/,  ptr /*devicePointer*/, sizeBytes, acc, flags));
                    } else {
                        g_amPointerTracker.insert(ptr, 
                                hc::AmPointerInfo(true/*isDevice*/, NULL/*hostPointer*/,  ptr /*devicePointer*/, sizeBytes, acc, flags));
                    }
                }
            }
        }
    }

    return ptr;
};


am_status_t AM_free(void* ptr) 
{
    am_status_t status = AM_SUCCESS;

    if (ptr != NULL) {
        hsa_memory_free(ptr);

        size_t numRemoved = g_amPointerTracker.remove(ptr) ;
        if (numRemoved == 0) {
            status = AM_ERROR_MISC;
        }
    }
    return status;
}



am_status_t AM_copy(void*  dst, const void*  src, size_t sizeBytes)
{
    am_status_t am_status = AM_ERROR_MISC;
    hsa_status_t err = hsa_memory_copy(dst, src, sizeBytes);

    if (err == HSA_STATUS_SUCCESS) {
        am_status = AM_SUCCESS;
    } else {
        am_status = AM_ERROR_MISC;
    }

    return am_status;
}


am_status_t AM_get_pointer_info(hc::AmPointerInfo *info, void *ptr)
{
    auto infoI = g_amPointerTracker.find(ptr);
    if (infoI != g_amPointerTracker.end()) {
        *info = infoI->second;
        return AM_SUCCESS;
    } else {
        return AM_ERROR_MISC;
    }
}

void AM_print_tracker()
{
    g_amPointerTracker.print(std::cerr);
}


} // end namespace hc.
