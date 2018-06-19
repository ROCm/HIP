// automatically generated sources
#ifndef _HIP_CBAPI_H
#define _HIP_CBAPI_H

#include <hip/hip_cbstr.h>

#include <atomic>
#include <mutex>

// Callbacks spawner instantiation
#define HIP_CALLBACKS_INSTANCE \
  hip_cb_table_t api_callbacks_spawner::callbacks_table{}; \
  api_callbacks_spawner::mutex_t api_callbacks_spawner::mutex_;

// Set HIP activity/callback macros
#define HIP_SET_ACTIVITY api_callbacks_spawner::set_activity
#define HIP_SET_CALLBACK api_callbacks_spawner::set_callback

// HIP API callbacks spawner object macro
#define CB_SPAWNER_OBJECT(CB_ID) \
  hip_cb_data_t cb_data{}; \
  INIT_CB_ARGS_DATA(CB_ID, cb_data); \
  api_callbacks_spawner __api_tracer(HIP_API_ID_##CB_ID, cb_data); 

// HIP API callbacks table
struct hip_cb_table_entry_t {
  volatile std::atomic<uint32_t> sem;
  volatile std::atomic<bool> sync;
  hip_cb_act_t act;
  void* a_arg;
  hip_cb_fun_t fun;
  void* arg;
};

struct hip_cb_table_t {
  hip_cb_table_entry_t arr[HIP_API_ID_NUMBER];
};

enum { HIP_DOMAIN_ID = 1 };

class api_callbacks_spawner {
 public:
  typedef std::recursive_mutex mutex_t;

  api_callbacks_spawner(const hip_cb_id_t& cid, hip_cb_data_t& cb_data) :
    cid_(cid),
    cb_data_(cb_data),
    record_(NULL)
  {
    if (cid >= HIP_API_ID_NUMBER) {
      fprintf(stderr, "HIP %s bad id %d\n", __FUNCTION__, (int)cid);
      abort();
    }
    sem_increment(cid);
    cb_act_ = callbacks_table.arr[cid].act;
    cb_a_arg_ = callbacks_table.arr[cid].a_arg;
    cb_fun_ = callbacks_table.arr[cid].fun;
    cb_arg_ = callbacks_table.arr[cid].arg;
    cb_data_.phase = 0;
    if (cb_act_ != NULL) cb_act_(cid, &record_, &cb_data_, cb_a_arg_);
    if (cb_fun_ != NULL) cb_fun_(HIP_DOMAIN_ID, cid, &cb_data_, cb_arg_);
  }

  ~api_callbacks_spawner() {
    cb_data_.phase = 1;
    if (cb_act_ != NULL) cb_act_(cid_, &record_, &cb_data_, cb_a_arg_);
    if (cb_fun_ != NULL) cb_fun_(HIP_DOMAIN_ID, cid_, &cb_data_, cb_arg_);
    sem_decrement(cid_);
  }

  static bool set_activity(uint32_t id, hip_cb_act_t fun, void* arg);
  static bool set_callback(uint32_t id, hip_cb_fun_t fun, void* arg);

 private:
  static void cb_sync(const uint32_t& id);
  static void cb_release(const uint32_t& id);
  static void sem_increment(const uint32_t& id);
  static void sem_decrement(const uint32_t& id);

  static hip_cb_table_t callbacks_table;
  static mutex_t mutex_;

  const hip_cb_id_t cid_;
  hip_cb_data_t& cb_data_;
  hip_act_record_t* record_;
  hip_cb_act_t cb_act_;
  void* cb_a_arg_;
  hip_cb_fun_t cb_fun_;
  void* cb_arg_;
};

inline void api_callbacks_spawner::cb_sync(const uint32_t& id) {
  callbacks_table.arr[id].sync.store(true);
  while (callbacks_table.arr[id].sem != 0) {}
}

inline void api_callbacks_spawner::cb_release(const uint32_t& id) {
  callbacks_table.arr[id].sync.store(false);
}

inline void api_callbacks_spawner::sem_increment(const uint32_t& id) {
  while (callbacks_table.arr[id].sync.load() == true) {}

  const uint32_t prev = callbacks_table.arr[id].sem.fetch_add(1);
  if (prev == UINT32_MAX) {
    std::cerr << "sem overflow id = " << id << std::endl << std::flush;
    abort();
  }
}

inline void api_callbacks_spawner::sem_decrement(const uint32_t& id) {
  const uint32_t prev = callbacks_table.arr[id].sem.fetch_sub(1);
  if (prev == 0) {
    std::cerr << "sem corrupted id = " << id << std::endl << std::flush;
    abort();
  }
}

inline bool api_callbacks_spawner::set_activity(uint32_t id, hip_cb_act_t fun, void* arg) {
  std::lock_guard<mutex_t> lock(mutex_);
  bool ret = true;
  if (id == HIP_API_ID_ANY) {
    for (unsigned i = 0; i < HIP_API_ID_NUMBER; ++i) set_activity(i, fun, arg);
  } else if (id < HIP_API_ID_NUMBER) {
    cb_sync(id);
    callbacks_table.arr[id].act = fun;
    callbacks_table.arr[id].a_arg = arg;
    cb_release(id);
  } else {
    ret = false;
  }
  return ret;
}

inline bool api_callbacks_spawner::set_callback(uint32_t id, hip_cb_fun_t fun, void* arg) {
  std::lock_guard<mutex_t> lock(mutex_);
  bool ret = true;
  if (id == HIP_API_ID_ANY) {
    for (unsigned i = 0; i < HIP_API_ID_NUMBER; ++i) set_callback(i, fun, arg);
  } else if (id < HIP_API_ID_NUMBER) {
    cb_sync(id);
    callbacks_table.arr[id].fun = fun;
    callbacks_table.arr[id].arg = arg;
    cb_release(id);
  } else {
    ret = false;
  }
  return ret;
}

#endif  // _HIP_CBAPI_H
