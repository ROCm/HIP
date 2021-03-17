/* Copyright (c) 2019-present Advanced Micro Devices, Inc.

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
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#ifndef HIP_SRC_HIP_PROF_API_H
#define HIP_SRC_HIP_PROF_API_H

#include <atomic>
#include <iostream>
#include <mutex>

#if USE_PROF_API
#include "hip/amd_detail/hip_prof_str.h"
#include "platform/prof_protocol.h"

// HIP API callbacks spawner object macro
#define HIP_CB_SPAWNER_OBJECT(CB_ID) \
  api_callbacks_spawner_t<HIP_API_ID_##CB_ID> __api_tracer; \
  { \
    hip_api_data_t* api_data = __api_tracer.get_api_data_ptr(); \
    if (api_data != NULL) { \
      hip_api_data_t& api_data_ref = *api_data; \
      INIT_CB_ARGS_DATA(CB_ID, api_data_ref); \
      __api_tracer.call(); \
    } \
  }

static const uint32_t HIP_DOMAIN_ID = ACTIVITY_DOMAIN_HIP_API;
typedef activity_record_t hip_api_record_t;
typedef activity_rtapi_callback_t hip_api_callback_t;
typedef activity_sync_callback_t hip_act_callback_t;

class api_callbacks_table_t {
 public:
  typedef std::mutex mutex_t;

  typedef hip_api_record_t record_t;
  typedef hip_api_callback_t fun_t;
  typedef hip_act_callback_t act_t;

  // HIP API callbacks table
  struct hip_cb_table_entry_t {
    volatile std::atomic<bool> sync;
    volatile std::atomic<uint32_t> sem;
    act_t act;
    void* a_arg;
    fun_t fun;
    void* arg;
  };

  struct hip_cb_table_t {
    hip_cb_table_entry_t arr[HIP_API_ID_NUMBER];
  };

  api_callbacks_table_t() {
     memset(&callbacks_table_, 0, sizeof(callbacks_table_));
  }

  bool set_activity(uint32_t id, act_t fun, void* arg) {
    std::lock_guard<mutex_t> lock(mutex_);
    bool ret = true;

    if (id < HIP_API_ID_NUMBER) {
      cb_sync(id);
      /*
      'fun != nullptr' indicates it is activity register call,
      increment should happen only once but client is free to call
      register CB multiple times for same API id hence the check

      'fun == nullptr' indicates it is de-register call and
      decrement should happen only once hence the check
      */
      if (fun != nullptr) {
        if (callbacks_table_.arr[id].act == nullptr) {
          enabled_api_count_++;
        }
      } else {
        if (callbacks_table_.arr[id].act != nullptr) {
          enabled_api_count_--;
        }
      }
      if (enabled_api_count_ > 0) {
        amd::IS_PROFILER_ON = true;
      } else {
        amd::IS_PROFILER_ON = false;
      }
      callbacks_table_.arr[id].act = fun;
      callbacks_table_.arr[id].a_arg = arg;
      cb_release(id);
    } else {
      ret = false;
    }

    return ret;
  }

  bool set_callback(uint32_t id, fun_t fun, void* arg) {
    std::lock_guard<mutex_t> lock(mutex_);
    bool ret = true;

    if (id < HIP_API_ID_NUMBER) {
      cb_sync(id);
      callbacks_table_.arr[id].fun = fun;
      callbacks_table_.arr[id].arg = arg;
      cb_release(id);
    } else {
      ret = false;
    }

    return ret;
  }

  void set_enabled(const bool& enabled) {
    amd::IS_PROFILER_ON = enabled;
  }

  inline hip_cb_table_entry_t& entry(const uint32_t& id) {
    return callbacks_table_.arr[id];
  }

  inline void sem_sync(const uint32_t& id) {
    sem_increment(id);
    if (entry(id).sync.load() == true) sync_wait(id);
  }

  inline void sem_release(const uint32_t& id) {
    sem_decrement(id);
  }

  inline bool is_enabled() const {
    return amd::IS_PROFILER_ON;
  }

 private:
  inline void cb_sync(const uint32_t& id) {
    entry(id).sync.store(true);
    while (entry(id).sem.load() != 0) {}
  }

  inline void cb_release(const uint32_t& id) {
    entry(id).sync.store(false);
  }

  inline void sem_increment(const uint32_t& id) {
    const uint32_t prev = entry(id).sem.fetch_add(1);
    if (prev == UINT32_MAX) {
      std::cerr << "sem overflow id = " << id << std::endl << std::flush;
      abort();
    }
  }

  inline void sem_decrement(const uint32_t& id) {
    const uint32_t prev = entry(id).sem.fetch_sub(1);
    if (prev == 0) {
      std::cerr << "sem corrupted id = " << id << std::endl << std::flush;
      abort();
    }
  }

  void sync_wait(const uint32_t& id) {
    sem_decrement(id);
    while (entry(id).sync.load() == true) {}
    sem_increment(id);
  }

  mutex_t mutex_;
  hip_cb_table_t callbacks_table_;
  uint32_t enabled_api_count_;
};

extern api_callbacks_table_t callbacks_table;

template <int cid_>
class api_callbacks_spawner_t {
 public:
  api_callbacks_spawner_t() :
    api_data_(NULL)
  {
    if (!is_enabled()) return;

    if (cid_ >= HIP_API_ID_NUMBER) {
      fprintf(stderr, "HIP %s bad id %d\n", __FUNCTION__, cid_);
      abort();
    }
    callbacks_table.sem_sync(cid_);

    hip_act_callback_t act = entry(cid_).act;
    if (act != NULL) api_data_ = (hip_api_data_t*) act(cid_, NULL, NULL, NULL);
  }

  void call() {
    hip_api_callback_t fun = entry(cid_).fun;
    void* arg = entry(cid_).arg;
    if (fun != NULL) {
      fun(HIP_DOMAIN_ID, cid_, api_data_, arg);
      api_data_->phase = ACTIVITY_API_PHASE_EXIT;
    }
  }

  ~api_callbacks_spawner_t() {
    if (!is_enabled()) return;

    if (api_data_ != NULL) {
      hip_api_callback_t fun = entry(cid_).fun;
      void* arg = entry(cid_).arg;
      hip_act_callback_t act = entry(cid_).act;
      void* a_arg = entry(cid_).a_arg;
      if (fun != NULL) fun(HIP_DOMAIN_ID, cid_, api_data_, arg);
      if (act != NULL) act(cid_, NULL, NULL, a_arg);
    }

    callbacks_table.sem_release(cid_);
  }

  hip_api_data_t* get_api_data_ptr() {
    return api_data_;
  }

  bool is_enabled() const {
    return callbacks_table.is_enabled();
  }

 private:
  inline api_callbacks_table_t::hip_cb_table_entry_t& entry(const uint32_t& id) {
    return callbacks_table.entry(id);
  }

  hip_api_data_t* api_data_;
};

template <>
class api_callbacks_spawner_t<HIP_API_ID_NUMBER> {
 public:
  api_callbacks_spawner_t() {}
  void call() {}
  hip_api_data_t* get_api_data_ptr() { return NULL; }
  bool is_enabled() const { return false; }
};

#else

#define HIP_CB_SPAWNER_OBJECT(x) do {} while(0)

class api_callbacks_table_t {
 public:
  typedef void* act_t;
  typedef void* fun_t;
  bool set_activity(uint32_t id, act_t fun, void* arg) { return false; }
  bool set_callback(uint32_t id, fun_t fun, void* arg) { return false; }
};

#endif

#endif  // HIP_SRC_HIP_PROF_API_H
