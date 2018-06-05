// automatically generated sources
#ifndef _HIP_CBAPI_H
#define _HIP_CBAPI_H

#include <hip/hip_cbstr.h>

#include <mutex>

// HIP API callbacks table
struct hip_cb_table_entry_t {
  hip_cb_act_t act;
  void* a_arg;
  hip_cb_fun_t fun;
  void* arg;
};
struct hip_cb_table_t {
  hip_cb_table_entry_t arr[HIP_API_ID_NUMBER];
};

#define HIP_CALLBACKS_INSTANCE \
  hip_cb_table_t HIP_API_callbacks_table{};
  
extern hip_cb_table_entry_t HIP_API_callbacks_global_entry;
extern hip_cb_table_t HIP_API_callbacks_table;

inline bool HIP_SET_ACTIVITY(uint32_t id, hip_cb_act_t fun, void* arg) {
  bool ret = true;
  if (id == 0) {
    for (unsigned i = 0; i < HIP_API_ID_NUMBER; ++i) {
      HIP_API_callbacks_table.arr[i].act = fun;
      HIP_API_callbacks_table.arr[i].a_arg = arg;
    }
  } else if (id < HIP_API_ID_NUMBER) {
    HIP_API_callbacks_table.arr[id].act = fun;
    HIP_API_callbacks_table.arr[id].a_arg = arg;
  } else {
    ret = false;
  }
  return ret;
}

inline bool HIP_SET_CALLBACK(uint32_t id, hip_cb_fun_t fun, void* arg) {
  bool ret = true;
  if (id == 0) {
    for (unsigned i = 0; i < HIP_API_ID_NUMBER; ++i) {
      HIP_API_callbacks_table.arr[i].fun = fun;
      HIP_API_callbacks_table.arr[i].arg = arg;
    }
  } else if (id < HIP_API_ID_NUMBER) {
    HIP_API_callbacks_table.arr[id].fun = fun;
    HIP_API_callbacks_table.arr[id].arg = arg;
  } else {
    ret = false;
  }
  return ret;
}

enum { HIP_DOMAIN_ID = 1 };

class api_callbacks_spawner_t {
 public:
  api_callbacks_spawner_t(const hip_cb_id_t& cid, hip_cb_data_t& cb_data) :
    cid_(cid),
    cb_data_(cb_data),
    record_(NULL)
  {
    if (cid >= HIP_API_ID_NUMBER) {
      fprintf(stderr, "HIP %s bad id %d\n", __FUNCTION__, (int)cid);
      abort();
    }
    cb_data_.phase = 0;
    cb_act_ = HIP_API_callbacks_table.arr[cid].act;
    cb_a_arg_ = HIP_API_callbacks_table.arr[cid].a_arg;
    cb_fun_ = HIP_API_callbacks_table.arr[cid].fun;
    cb_arg_ = HIP_API_callbacks_table.arr[cid].arg;
    if (cb_act_ != NULL) cb_act_(cid_, &record_, &cb_data_, cb_a_arg_);
    if (cb_fun_ != NULL) cb_fun_(HIP_DOMAIN_ID, cid_, &cb_data_, cb_arg_);
  }
  ~api_callbacks_spawner_t() {
    cb_data_.phase = 1;
    if (cb_act_ != NULL) cb_act_(cid_, &record_, &cb_data_, cb_a_arg_);
    if (cb_fun_ != NULL) cb_fun_(HIP_DOMAIN_ID, cid_, &cb_data_, cb_arg_);
  }
 private:
  const hip_cb_id_t cid_;
  hip_cb_data_t& cb_data_;
  hip_act_record_t* record_;
  hip_cb_act_t cb_act_;
  void* cb_a_arg_;
  hip_cb_fun_t cb_fun_;
  void* cb_arg_;
};

// HIP API callbacks spawning class macro
#define CB_SPAWNER_OBJECT(CB_ID) \
  hip_cb_data_t cb_data{}; \
  INIT_CB_ARGS_DATA(CB_ID, cb_data); \
  api_callbacks_spawner_t __api_tracer(HIP_API_ID_##CB_ID, cb_data); 

#endif  // _HIP_CBAPI_H
