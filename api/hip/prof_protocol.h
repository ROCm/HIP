/*
Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.

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

#include <stdlib.h>

// Traced API domains
typedef enum {
  ACTIVITY_DOMAIN_HSA_API = 0,                    // HSA API domain
  ACTIVITY_DOMAIN_HSA_OPS = 1,                    // HSA async activity domain
  ACTIVITY_DOMAIN_HCC_OPS = 2,                    // HCC async activity domain
  ACTIVITY_DOMAIN_HIP_API = 3,                    // HIP API domain
  ACTIVITY_DOMAIN_EXT_API = 4,                    // External ID domain
  ACTIVITY_DOMAIN_ROCTX   = 5,                    // ROCTX domain
  ACTIVITY_DOMAIN_NUMBER
} activity_domain_t;

// Extension API opcodes
typedef enum {
  ACTIVITY_EXT_OP_MARK = 0,
  ACTIVITY_EXT_OP_EXTERN_ID = 1
} activity_ext_op_t;

// API calback type
typedef void (*activity_rtapi_callback_t)(uint32_t domain, uint32_t cid, const void* data, void* arg);
typedef uint32_t activity_kind_t;
typedef uint32_t activity_op_t;

// API callback phase
typedef enum {
  ACTIVITY_API_PHASE_ENTER = 0,
  ACTIVITY_API_PHASE_EXIT = 1
} activity_api_phase_t;

// Trace record types
// Correlation id
typedef uint64_t activity_correlation_id_t;

// Activity record type
struct activity_record_t {
    uint32_t domain;                               // activity domain id
    activity_kind_t kind;                          // activity kind
    activity_op_t op;                              // activity op
    activity_correlation_id_t correlation_id;      // activity ID
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
      struct {
        activity_correlation_id_t external_id;     // external correlatino id
      };
    };
    size_t bytes;                                  // data size bytes
};

// Activity sync calback type
typedef void* (*activity_sync_callback_t)(uint32_t cid, activity_record_t* record, const void* data, void* arg);
// Activity async calback type
typedef void (*activity_id_callback_t)(activity_correlation_id_t id);
typedef void (*activity_async_callback_t)(uint32_t op, void* record, void* arg);

#endif  // INC_EXT_PROF_PROTOCOL_H_
