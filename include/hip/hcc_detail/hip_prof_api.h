// automatically generated sources
#ifndef _HIP_PROF_API_H
#define _HIP_PROF_API_H

#include <atomic>
#include <iostream>
#include <mutex>

#include "hip/hcc_detail/hip_prof_str.h"

template <typename Record, typename Fun, typename Act>
class api_callbacks_table_templ {
 public:
  typedef std::recursive_mutex mutex_t;

  typedef Record record_t;
  typedef Fun fun_t;
  typedef Act act_t;

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

  api_callbacks_table_templ() {
     memset(&callbacks_table_, 0, sizeof(callbacks_table_));
  }

  bool set_activity(uint32_t id, act_t fun, void* arg) {
    std::lock_guard<mutex_t> lock(mutex_);
    bool ret = true;
    if (id == HIP_API_ID_ANY) {
      for (unsigned i = 0; i < HIP_API_ID_NUMBER; ++i) set_activity(i, fun, arg);
    } else if (id < HIP_API_ID_NUMBER) {
      cb_sync(id);
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
    if (id == HIP_API_ID_ANY) {
      for (unsigned i = 0; i < HIP_API_ID_NUMBER; ++i) set_callback(i, fun, arg);
    } else if (id < HIP_API_ID_NUMBER) {
      cb_sync(id);
      callbacks_table_.arr[id].fun = fun;
      callbacks_table_.arr[id].arg = arg;
      cb_release(id);
    } else {
      ret = false;
    }
    return ret;
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
};


#if USE_PROF_API
#include <prof_protocol.h>

static const uint32_t HIP_DOMAIN_ID = ACTIVITY_DOMAIN_HIP_API;
typedef activity_record_t hip_api_record_t;
typedef activity_rtapi_callback_t hip_api_callback_t;
typedef activity_sync_callback_t hip_act_callback_t;

// HIP API callbacks spawner object macro
#define HIP_CB_SPAWNER_OBJECT(CB_ID) \
  hip_api_data_t api_data{}; \
  INIT_CB_ARGS_DATA(CB_ID, api_data); \
  api_callbacks_spawner_t<HIP_API_ID_##CB_ID> __api_tracer(HIP_API_ID_##CB_ID, api_data);

typedef api_callbacks_table_templ<hip_api_record_t,
                                  hip_api_callback_t,
                                  hip_act_callback_t> api_callbacks_table_t;
extern api_callbacks_table_t callbacks_table;

template <int cid_>
class api_callbacks_spawner_t {
 public:
  api_callbacks_spawner_t(const hip_api_id_t& cid, hip_api_data_t& api_data) :
    api_data_(api_data),
    record_({})
  {
    if (cid_ >= HIP_API_ID_NUMBER) {
      fprintf(stderr, "HIP %s bad id %d\n", __FUNCTION__, cid_);
      abort();
    }
    callbacks_table.sem_sync(cid_);

    act = entry(cid_).act;
    a_arg = entry(cid_).a_arg;
    fun = entry(cid_).fun;
    arg = entry(cid_).arg;

    api_data_.phase = 0;
    if (act != NULL) act(cid_, &record_, &api_data_, a_arg);
    if (fun != NULL) fun(HIP_DOMAIN_ID, cid_, &api_data_, arg);
  }

  ~api_callbacks_spawner_t() {
    api_data_.phase = 1;
    if (act != NULL) act(cid_, &record_, &api_data_, a_arg);
    if (fun != NULL) fun(HIP_DOMAIN_ID, cid_, &api_data_, arg);

    callbacks_table.sem_release(cid_);
  }

 private:
  inline api_callbacks_table_t::hip_cb_table_entry_t& entry(const uint32_t& id) {
    return callbacks_table.entry(id);
  }

  hip_api_data_t& api_data_;
  hip_api_record_t record_;

  hip_act_callback_t act;
  void* a_arg;
  hip_api_callback_t fun;
  void* arg;
};

template <>
class api_callbacks_spawner_t<HIP_API_ID_NUMBER> {
 public:
  api_callbacks_spawner_t(const hip_api_id_t& cid, hip_api_data_t& api_data) {}
};

#else

#define HIP_CB_SPAWNER_OBJECT(x) do {} while(0)

class api_callbacks_table_t {
 public:
  typedef void* act_t;
  typedef void* fun_t;
  bool set_activity(uint32_t id, act_t fun, void* arg) { return true; }
  bool set_callback(uint32_t id, fun_t fun, void* arg) { return true; }
};

#endif

#endif  // _HIP_PROF_API_H
