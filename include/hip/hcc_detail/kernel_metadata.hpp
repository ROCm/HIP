#ifndef KERNEL_METADATA_HPP
#define KERNEL_METADATA_HPP

#include <map>
#include <amd_comgr.h>

namespace hip_impl {

//  for Code Object V2
enum class AttrField : uint8_t {
  ReqdWorkGroupSize  = 0,
  WorkGroupSizeHint = 1,
  VecTypeHint       = 2,
  RuntimeHandle     = 3
};

enum class CodePropField : uint8_t {
  KernargSegmentSize      = 0,
  GroupSegmentFixedSize   = 1,
  PrivateSegmentFixedSize = 2,
  KernargSegmentAlign     = 3,
  WavefrontSize           = 4,
  NumSGPRs                = 5,
  NumVGPRs                = 6,
  MaxFlatWorkGroupSize    = 7,
  IsDynamicCallStack      = 8,
  IsXNACKEnabled          = 9,
  NumSpilledSGPRs         = 10,
  NumSpilledVGPRs         = 11
};

static const std::map<std::string,AttrField> AttrFieldMap =
{
  {"ReqdWorkGroupSize",   AttrField::ReqdWorkGroupSize},
  {"WorkGroupSizeHint",   AttrField::WorkGroupSizeHint},
  {"VecTypeHint",         AttrField::VecTypeHint},
  {"RuntimeHandle",       AttrField::RuntimeHandle}
};

static const std::map<std::string,CodePropField> CodePropFieldMap =
{
  {"KernargSegmentSize",      CodePropField::KernargSegmentSize},
  {"GroupSegmentFixedSize",   CodePropField::GroupSegmentFixedSize},
  {"PrivateSegmentFixedSize", CodePropField::PrivateSegmentFixedSize},
  {"KernargSegmentAlign",     CodePropField::KernargSegmentAlign},
  {"WavefrontSize",           CodePropField::WavefrontSize},
  {"NumSGPRs",                CodePropField::NumSGPRs},
  {"NumVGPRs",                CodePropField::NumVGPRs},
  {"MaxFlatWorkGroupSize",    CodePropField::MaxFlatWorkGroupSize},
  {"IsDynamicCallStack",      CodePropField::IsDynamicCallStack},
  {"IsXNACKEnabled",          CodePropField::IsXNACKEnabled},
  {"NumSpilledSGPRs",         CodePropField::NumSpilledSGPRs},
  {"NumSpilledVGPRs",         CodePropField::NumSpilledVGPRs}
};

/// In-memory representation of kernel metadata.
struct KernelMD {
  /// Name of the kernel descriptor ELF symbol
  std::string mSymbolName = std::string();
  /// 'reqd_work_group_size' attribute. Optional.
  std::vector<uint32_t> mReqdWorkGroupSize = std::vector<uint32_t>();
  /// 'work_group_size_hint' attribute. Optional.
  std::vector<uint32_t> mWorkGroupSizeHint = std::vector<uint32_t>();
  /// 'vec_type_hint' attribute. Optional.
  std::string mVecTypeHint = std::string();
  /// External symbol created by runtime to store the kernel address
  /// for enqueued blocks.
  std::string mRuntimeHandle = std::string();
  /// Size in bytes of the kernarg segment memory. Kernarg segment memory
  /// holds the values of the arguments to the kernel. Required.
  uint64_t mKernargSegmentSize = 0;
  /// Size in bytes of the group segment memory required by a workgroup.
  /// This value does not include any dynamically allocated group segment memory
  /// that may be added when the kernel is dispatched. Required.
  uint32_t mGroupSegmentFixedSize = 0;
  /// Size in bytes of the private segment memory required by a workitem.
  /// Private segment memory includes arg, spill and private segments. Required.
  uint32_t mPrivateSegmentFixedSize = 0;
  /// Maximum byte alignment of variables used by the kernel in the
  /// kernarg memory segment. Required.
  uint32_t mKernargSegmentAlign = 0;
  /// Wavefront size. Required.
  uint32_t mWavefrontSize = 0;
  /// Total number of SGPRs used by a wavefront. Optional.
  uint16_t mNumSGPRs = 0;
  /// Total number of VGPRs used by a workitem. Optional.
  uint16_t mNumVGPRs = 0;
  /// Maximum flat work-group size supported by the kernel. Optional.
  uint32_t mMaxFlatWorkGroupSize = 0;
  /// True if the generated machine code is using a dynamically sized
  /// call stack. Optional.
  bool mIsDynamicCallStack = false;
  /// True if the generated machine code is capable of supporting XNACK.
  /// Optional.
  bool mIsXNACKEnabled = false;
  /// Number of SGPRs spilled by a wavefront. Optional.
  uint16_t mNumSpilledSGPRs = 0;
  /// Number of VGPRs spilled by a workitem. Optional.
  uint16_t mNumSpilledVGPRs = 0;
};

struct kernelInfo {
  KernelMD descriptor;        /// attributes and code properties
  /// Vector of kernel argument information: size, align (for v2)
  std::vector<std::pair<std::size_t, std::size_t>> args_info;
};

static amd_comgr_status_t getMetaBuf(const amd_comgr_metadata_node_t meta,
                                     std::string* str) {
  size_t size = 0;
  amd_comgr_status_t status = amd_comgr_get_metadata_string(meta, &size, NULL);

  if (status == AMD_COMGR_STATUS_SUCCESS) {
    str->resize(size-1);    // minus one to discount the null character
    status = amd_comgr_get_metadata_string(meta, &size, &((*str)[0]));
  }

  return status;
}

static amd_comgr_status_t populateAttrs(const amd_comgr_metadata_node_t key,
                                        const amd_comgr_metadata_node_t value,
                                        void *data) {
  amd_comgr_status_t status;
  amd_comgr_metadata_kind_t kind;
  size_t size = 0;
  std::string buf;

  // get the key of the argument field
  status = amd_comgr_get_metadata_kind(key, &kind);
  if (kind == AMD_COMGR_METADATA_KIND_STRING && status == AMD_COMGR_STATUS_SUCCESS) {
    status = getMetaBuf(key, &buf);
  }

  if (status != AMD_COMGR_STATUS_SUCCESS) {
    return AMD_COMGR_STATUS_ERROR;
  }

  auto itAttrField = AttrFieldMap.find(buf);
  if (itAttrField == AttrFieldMap.end()) {
    return AMD_COMGR_STATUS_ERROR;
  }

  KernelMD* kernelMD = static_cast<KernelMD*>(data);
  switch (itAttrField->second) {
    case AttrField::ReqdWorkGroupSize:
      {
        status = amd_comgr_get_metadata_list_size(value, &size);
        if (size == 3 && status == AMD_COMGR_STATUS_SUCCESS) {
          for (size_t i = 0; i < size && status == AMD_COMGR_STATUS_SUCCESS; i++) {
            amd_comgr_metadata_node_t workgroupSize;
            status = amd_comgr_index_list_metadata(value, i, &workgroupSize);

            if (status == AMD_COMGR_STATUS_SUCCESS &&
                getMetaBuf(workgroupSize, &buf) == AMD_COMGR_STATUS_SUCCESS) {
              kernelMD->mReqdWorkGroupSize.push_back(atoi(buf.c_str()));
            }
            amd_comgr_destroy_metadata(workgroupSize);
          }
        }
      }
      break;
    case AttrField::WorkGroupSizeHint:
      {
        status = amd_comgr_get_metadata_list_size(value, &size);
        if (status == AMD_COMGR_STATUS_SUCCESS && size == 3) {
          for (size_t i = 0; i < size && status == AMD_COMGR_STATUS_SUCCESS; i++) {
            amd_comgr_metadata_node_t workgroupSizeHint;
            status = amd_comgr_index_list_metadata(value, i, &workgroupSizeHint);

            if (status == AMD_COMGR_STATUS_SUCCESS &&
                getMetaBuf(workgroupSizeHint, &buf) == AMD_COMGR_STATUS_SUCCESS) {
              kernelMD->mWorkGroupSizeHint.push_back(atoi(buf.c_str()));
            }
            amd_comgr_destroy_metadata(workgroupSizeHint);
          }
        }
      }
      break;
    case AttrField::VecTypeHint:
      {
        if (getMetaBuf(value,&buf) == AMD_COMGR_STATUS_SUCCESS) {
          kernelMD->mVecTypeHint = buf;
        }
      }
      break;
    case AttrField::RuntimeHandle:
      {
        if (getMetaBuf(value,&buf) == AMD_COMGR_STATUS_SUCCESS) {
          kernelMD->mRuntimeHandle = buf;
        }
      }
      break;
    default:
      return AMD_COMGR_STATUS_ERROR;
  }

  return status;
}

static amd_comgr_status_t populateCodeProps(const amd_comgr_metadata_node_t key,
                                            const amd_comgr_metadata_node_t value,
                                            void *data) {
  amd_comgr_status_t status;
  amd_comgr_metadata_kind_t kind;
  std::string buf;

  // get the key of the argument field
  status = amd_comgr_get_metadata_kind(key, &kind);
  if (kind == AMD_COMGR_METADATA_KIND_STRING && status == AMD_COMGR_STATUS_SUCCESS) {
    status = getMetaBuf(key, &buf);
  }

  if (status != AMD_COMGR_STATUS_SUCCESS) {
    return AMD_COMGR_STATUS_ERROR;
  }

  auto itCodePropField = CodePropFieldMap.find(buf);
  if (itCodePropField == CodePropFieldMap.end()) {
    return AMD_COMGR_STATUS_ERROR;
  }

  // get the value of the argument field
  if (status == AMD_COMGR_STATUS_SUCCESS) {
    status = getMetaBuf(value, &buf);
  }

  KernelMD*  kernelMD = static_cast<KernelMD*>(data);
  switch (itCodePropField->second) {
    case CodePropField::KernargSegmentSize:
      kernelMD->mKernargSegmentSize = atoi(buf.c_str());
      break;
    case CodePropField::GroupSegmentFixedSize:
      kernelMD->mGroupSegmentFixedSize = atoi(buf.c_str());
      break;
    case CodePropField::PrivateSegmentFixedSize:
      kernelMD->mPrivateSegmentFixedSize = atoi(buf.c_str());
      break;
    case CodePropField::KernargSegmentAlign:
      kernelMD->mKernargSegmentAlign = atoi(buf.c_str());
      break;
    case CodePropField::WavefrontSize:
      kernelMD->mWavefrontSize = atoi(buf.c_str());
      break;
    case CodePropField::NumSGPRs:
      kernelMD->mNumSGPRs = atoi(buf.c_str());
      break;
    case CodePropField::NumVGPRs:
      kernelMD->mNumVGPRs = atoi(buf.c_str());
      break;
    case CodePropField::MaxFlatWorkGroupSize:
      kernelMD->mMaxFlatWorkGroupSize = atoi(buf.c_str());
      break;
    case CodePropField::IsDynamicCallStack:
        kernelMD->mIsDynamicCallStack = (buf.compare("true") == 0);
      break;
    case CodePropField::IsXNACKEnabled:
      kernelMD->mIsXNACKEnabled = (buf.compare("true") == 0);
      break;
    case CodePropField::NumSpilledSGPRs:
      kernelMD->mNumSpilledSGPRs = atoi(buf.c_str());
      break;
    case CodePropField::NumSpilledVGPRs:
      kernelMD->mNumSpilledVGPRs = atoi(buf.c_str());
      break;
    default:
      return AMD_COMGR_STATUS_ERROR;
  }
  return AMD_COMGR_STATUS_SUCCESS;
}
}  // Namespace hip_impl.

#endif // KERNEL_METADATA_HPP
