#ifndef HIP_FAT_BINARY_HPP
#define HIP_FAT_BINARY_HPP

#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#include "hip_internal.hpp"
#include "platform/program.hpp"

namespace hip {

//Fat Binary Per Device info
class FatBinaryDeviceInfo {
public:
  FatBinaryDeviceInfo (const void* binary_image, size_t binary_size, size_t binary_offset)
                      : binary_image_(binary_image), binary_size_(binary_size),
                        binary_offset_(binary_offset), program_(nullptr),
                        add_dev_prog_(false), prog_built_(false) {}

  ~FatBinaryDeviceInfo();

private:
  const void* binary_image_; // binary image ptr
  size_t binary_size_;       // binary image size
  size_t binary_offset_;     // image offset from original

  amd::Program* program_;    // reinterpreted as hipModule_t
  friend class FatBinaryInfo;

  //Control Variables
  bool add_dev_prog_;
  bool prog_built_;
};


// Fat Binary Info
class FatBinaryInfo {
public:
  FatBinaryInfo(const char* fname, const void* image);
  ~FatBinaryInfo();

  // Loads Fat binary from file or image, unbundles COs for devices.
  hipError_t ExtractFatBinary(const std::vector<hip::Device*>& devices);
  hipError_t AddDevProgram(const int device_id);
  hipError_t BuildProgram(const int device_id);


  // Device Id bounds check
  inline void DeviceIdCheck(const int device_id) const {
    guarantee(device_id >= 0, "Invalid DeviceId less than 0");
    guarantee(static_cast<size_t>(device_id) < fatbin_dev_info_.size(), "Invalid DeviceId, greater than no of fatbin device info!");
  }

  // Getter Methods
  amd::Program* GetProgram(int device_id) {
    DeviceIdCheck(device_id);
    return fatbin_dev_info_[device_id]->program_;
  }

  hipModule_t Module(int device_id) const {
    DeviceIdCheck(device_id);
    return reinterpret_cast<hipModule_t>(as_cl(fatbin_dev_info_[device_id]->program_));
  }

  hipError_t GetModule(int device_id, hipModule_t* hmod) const {
    DeviceIdCheck(device_id);
    *hmod = reinterpret_cast<hipModule_t>(as_cl(fatbin_dev_info_[device_id]->program_));
    return hipSuccess;
  }

private:
  std::string fname_;        // File name
  amd::Os::FileDesc fdesc_;  // File descriptor
  size_t fsize_;             // Total file size

  // Even when file is passed image will be mmapped till ~desctructor.
  const void* image_;        // Image

  // Only used for FBs where image is directly passed
  std::string uri_;          // Uniform resource indicator

  // Per Device Info, like corresponding binary ptr, size.
  std::vector<FatBinaryDeviceInfo*> fatbin_dev_info_;
};

}; /* namespace hip */

#endif /* HIP_FAT_BINARY_HPP */
