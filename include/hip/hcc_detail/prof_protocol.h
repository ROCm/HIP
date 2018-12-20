/*
Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef INC_EXT_PROF_PROTOCOL_H_
#define INC_EXT_PROF_PROTOCOL_H_

// Traced API domains
typedef enum {
  ACTIVITY_DOMAIN_ANY = 0,                        // Any domain
  ACTIVITY_DOMAIN_HSA_API = 1,                    // HSA domain
  ACTIVITY_DOMAIN_HCC_OPS = 2,                    // HCC domain
  ACTIVITY_DOMAIN_HIP_API = 3,                    // HIP domain
  ACTIVITY_DOMAIN_NUMBER = 4
} activity_domain_t;

// API calback type
typedef void (*activity_rtapi_callback_t)(uint32_t domain, uint32_t cid, const void* data, void* arg);

// API callback phase
typedef enum {
  ACTIVITY_API_PHASE_ENTER = 0,
  ACTIVITY_API_PHASE_EXIT = 1
} r_feature_kind_t;

// Trace record types
// Correlation id
typedef uint64_t activity_correlation_id_t;

// Activity record type
struct activity_record_t {
    uint32_t domain;                               // activity domain id
    uint32_t kind;                                 // activity kind
    uint32_t activity_id;                          // activity id
    activity_correlation_id_t correlation_id;      // activity correlation ID
    uint64_t begin_ns;                             // host begin timestamp
    uint64_t end_ns;                               // host end timestamp
    union {
      struct {
        int device_id;                             // device id
        uint64_t queue_id;                         // queue id
      };
      struct {
        uint32_t process_id;                       // device id
        uint32_t thread_id;                        // thread id
      };
    };
    size_t bytes;                                  // data size bytes
};

// Activity sync calback type
typedef activity_record_t* (*activity_sync_callback_t)(uint32_t cid, activity_record_t* record, const void* data, void* arg);
// Activity async calback type
typedef void (*activity_async_callback_t)(uint32_t op, void* record, void* arg);

#endif  // INC_EXT_PROF_PROTOCOL_H_
