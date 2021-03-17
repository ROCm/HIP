#pragma once

#ifdef __unix__

#include <string>
#include <atomic>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>

template <typename T>
struct Shmem {
  std::atomic<T> handle_;
  std::atomic<int> done_counter_;
};

template <typename T>
struct ShmemMeta {
  std::string shmem_name_;
  int shmem_fd_;
  Shmem<T>* shmem_;
};

template <typename T>
class MultiProcess {
public:
  MultiProcess(size_t num_proc) : num_proc_(num_proc) {}
  ~MultiProcess();

  void DebugInfo(pid_t pid);

  pid_t SpawnProcess(bool debug_bkpt);
  bool CreateShmem();

  bool WriteHandleToShmem(T ipc_handle);
  bool WaitTillAllChildReads();

  bool ReadHandleFromShmem(T& ipc_handle);
  bool NotifyParentDone();

private:
  const size_t num_proc_;
  bool debug_proc_;
  ShmemMeta<T> shmem_meta_obj_;
};

// Template Implementations
template <typename T>
MultiProcess<T>::~MultiProcess() {
  if(munmap(shmem_meta_obj_.shmem_, sizeof(Shmem<T>)) < 0) {
    std::cout<<"Error Unmapping shared memory "<<std::endl;
    exit(0);
  }
}

template <typename T>
void MultiProcess<T>::DebugInfo(pid_t pid) {
  const int delay = 1;

  if (pid == 0) {
    std::cout<<" Child Process with ID: "<<getpid()<<std::endl;
  } else {
    std::cout<<" Parent Process with ID: "<<getpid()<<std::endl;
  }

  volatile int flag = 0;
  while (!flag) {
    sleep(delay);
  }
}

template <typename T>
pid_t MultiProcess<T>::SpawnProcess(bool debug_bkpt) {
  if (num_proc_ < 0) {
    std::cout<<"Num Process cannot be less than 1"<<std::endl;
    return -1;
  }

  pid_t pid;
  for (size_t proc_idx = 0; proc_idx < num_proc_; ++proc_idx) {
    pid = fork();
    if (pid < 0) {
      std::cout<<"Fork Failed"<<std::endl;
      assert(false);
    } else if (pid == 0) {
      //Child Process, so break
      break;
    }
  }

  if (debug_bkpt) {
    DebugInfo(pid);
  }

  return pid;
}

template <typename T>
bool MultiProcess<T>::CreateShmem() {
  if (num_proc_ < 0) {
    std::cout<<"Num Process cannot be less than 1"<<std::endl;
    return false;
  }

  char name_template[] = "/tmp/eventXXXXX";
  int temp_fd = mkstemp(name_template);
  shmem_meta_obj_.shmem_name_ = name_template;
  shmem_meta_obj_.shmem_name_.replace(0, 5, "/hip_");
  shmem_meta_obj_.shmem_fd_ = shm_open(shmem_meta_obj_.shmem_name_.c_str(),
                                       O_RDWR | O_CREAT, 0777);

  if (ftruncate(shmem_meta_obj_.shmem_fd_, sizeof(ShmemMeta<T>)) != 0) {
    std::cout<<"Cannot FTruncate "<<std::endl;
    exit(0);
  }

  shmem_meta_obj_.shmem_ = (Shmem<T>*)mmap(0, sizeof(Shmem<T>), PROT_READ | PROT_WRITE,
                                           MAP_SHARED, shmem_meta_obj_.shmem_fd_, 0);
  memset(&shmem_meta_obj_.shmem_->handle_, 0x00, sizeof(T));
  shmem_meta_obj_.shmem_->done_counter_ = -1;

  return true;
}

template <typename T>
bool MultiProcess<T>::WriteHandleToShmem(T ipc_handle) {
  memcpy(&shmem_meta_obj_.shmem_->handle_, &ipc_handle, sizeof(T));
  shmem_meta_obj_.shmem_->done_counter_ = 0;
  return true;
}

template <typename T>
bool MultiProcess<T>::WaitTillAllChildReads() {
  size_t write_count = 0;
  while (shmem_meta_obj_.shmem_->done_counter_ != num_proc_) {
    ++write_count;
  }
  return true;
}

template <typename T>
bool MultiProcess<T>::ReadHandleFromShmem(T& ipc_handle) {
  size_t read_count = 0;
  while (shmem_meta_obj_.shmem_->done_counter_ == -1) {
    ++read_count;
  }
  memcpy(&ipc_handle, &shmem_meta_obj_.shmem_->handle_, sizeof(T));
  return true;
}

template <typename T>
bool MultiProcess<T>::NotifyParentDone() {
  ++shmem_meta_obj_.shmem_->done_counter_;
  return true;
}

#endif /* __unix__ */
