#include "hip_fatbin.hpp"

#include "hip_code_object.hpp"

namespace hip {

FatBinaryDeviceInfo::~FatBinaryDeviceInfo() {
  if (program_ != nullptr) {
    program_->release();
    program_ = nullptr;
  }
}

FatBinaryInfo::FatBinaryInfo(const char* fname, const void* image)
               : fdesc_(amd::Os::FDescInit()), fsize_(0), image_(image), uri_(std::string()) {

  if (fname != nullptr) {
    fname_ = std::string(fname);
  } else {
    fname_ = std::string();
  }

  fatbin_dev_info_.resize(g_devices.size());
}

FatBinaryInfo::~FatBinaryInfo() {

  for (auto& fbd: fatbin_dev_info_) {
    delete fbd;
  }

  if (fdesc_ > 0) {
    if (fsize_ && !amd::Os::MemoryUnmapFile(image_, fsize_)) {
      guarantee(false, "Cannot unmap file");
    }
    if (!amd::Os::CloseFileHandle(fdesc_)) {
      guarantee(false, "Cannot close file");
    }
  }

  fname_ = std::string();
  fdesc_ = amd::Os::FDescInit();
  fsize_ = 0;
  image_ = nullptr;
  uri_ = std::string();
}

hipError_t FatBinaryInfo::ExtractFatBinary(const std::vector<hip::Device*>& devices) {
  hipError_t hip_error = hipSuccess;
  std::vector<std::pair<const void*, size_t>> code_objs;

  // Copy device names for Extract Code object File
  std::vector<std::string> device_names;
  device_names.reserve(devices.size());
  for (size_t dev_idx = 0; dev_idx < devices.size(); ++dev_idx) {
    device_names.push_back(devices[dev_idx]->devices()[0]->isa().isaName());
  }

  // We are given file name, get the file desc and file size
  if (fname_.size() > 0) {
    // Get File Handle & size of the file.
    if (!amd::Os::GetFileHandle(fname_.c_str(), &fdesc_, &fsize_)) {
      return hipErrorFileNotFound;
    }
    if (fsize_ == 0) {
      return hipErrorInvalidKernelFile;
    }

    // Extract the code object from file
    hip_error = CodeObject::ExtractCodeObjectFromFile(fdesc_, fsize_, &image_,
                device_names, code_objs);

  } else if (image_ != nullptr) {
    // We are directly given image pointer directly, try to extract file desc & file Size
    hip_error = CodeObject::ExtractCodeObjectFromMemory(image_,
                device_names, code_objs, uri_);
  } else {
    return hipErrorInvalidValue;
  }

  if (hip_error == hipErrorNoBinaryForGpu) {
    guarantee(false, "hipErrorNoBinaryForGpu: Couldn't find binary for current devices!");
    return hip_error;
  }

  if (hip_error == hipErrorInvalidKernelFile) {
    for (size_t dev_idx = 0; dev_idx < devices.size(); ++dev_idx) {
      // the image type is no CLANG_OFFLOAD_BUNDLER, image for current device directly passed
      fatbin_dev_info_[devices[dev_idx]->deviceId()]
        = new FatBinaryDeviceInfo(image_, CodeObject::ElfSize(image_), 0);
    }
  } else if(hip_error == hipSuccess) {
    for (size_t dev_idx = 0; dev_idx < devices.size(); ++dev_idx) {
      // Calculate the offset wrt binary_image and the original image
      size_t offset_l
        = (reinterpret_cast<address>(const_cast<void*>(code_objs[dev_idx].first))
            - reinterpret_cast<address>(const_cast<void*>(image_)));

      fatbin_dev_info_[devices[dev_idx]->deviceId()]
        = new FatBinaryDeviceInfo(code_objs[dev_idx].first, code_objs[dev_idx].second, offset_l);
    }
  }

  for (size_t dev_idx = 0; dev_idx < devices.size(); ++dev_idx) {
    fatbin_dev_info_[devices[dev_idx]->deviceId()]->program_
       = new amd::Program(*devices[dev_idx]->asContext());
    if (fatbin_dev_info_[devices[dev_idx]->deviceId()]->program_ == NULL) {
      return hipErrorOutOfMemory;
    }
  }

  return hipSuccess;
}

hipError_t FatBinaryInfo::AddDevProgram(const int device_id) {
  // Device Id bounds Check
  DeviceIdCheck(device_id);

  FatBinaryDeviceInfo* fbd_info = fatbin_dev_info_[device_id];
  // If fat binary was already added, skip this step and return success
  if (fbd_info->add_dev_prog_ == false) {
    amd::Context* ctx = g_devices[device_id]->asContext();
    if (CL_SUCCESS != fbd_info->program_->addDeviceProgram(*ctx->devices()[0],
                                          fbd_info->binary_image_,
                                          fbd_info->binary_size_, false,
                                          nullptr, nullptr, fdesc_,
                                          fbd_info->binary_offset_, uri_)) {
      return hipErrorInvalidKernelFile;
    }
    fbd_info->add_dev_prog_ = true;
  }
  return hipSuccess;
}

hipError_t FatBinaryInfo::BuildProgram(const int device_id) {

  // Device Id Check and Add DeviceProgram if not added so far
  DeviceIdCheck(device_id);
  IHIP_RETURN_ONFAIL(AddDevProgram(device_id));

  // If Program was already built skip this step and return success
  FatBinaryDeviceInfo* fbd_info = fatbin_dev_info_[device_id];
  if (fbd_info->prog_built_ == false) {
    if(CL_SUCCESS != fbd_info->program_->build(g_devices[device_id]->devices(),
                                               nullptr, nullptr, nullptr,
                                               kOptionChangeable, kNewDevProg)) {
      return hipErrorSharedObjectInitFailed;
    }
    fbd_info->prog_built_ = true;
  }

  if (!fbd_info->program_->load()) {
    return hipErrorSharedObjectInitFailed;
  }
  return hipSuccess;
}

} //namespace : hip
