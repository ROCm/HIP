hipError_t ihipMemcpy3D_validate(const hipMemcpy3DParms* p);

hipError_t ihipMemcpy_validate(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind);

hipError_t ihipMemcpyCommand(amd::Command*& command, void* dst, const void* src, size_t sizeBytes,
                             hipMemcpyKind kind, amd::HostQueue& queue);

hipError_t ihipLaunchKernel_validate(hipFunction_t f, uint32_t globalWorkSizeX,
                                     uint32_t globalWorkSizeY, uint32_t globalWorkSizeZ,
                                     uint32_t blockDimX, uint32_t blockDimY, uint32_t blockDimZ,
                                     uint32_t sharedMemBytes, void** kernelParams, void** extra,
                                     int deviceId, uint32_t params);

hipError_t ihipMemset_validate(void* dst, int64_t value, size_t valueSize, size_t sizeBytes);

hipError_t ihipMemset3D_validate(hipPitchedPtr pitchedDevPtr, int value, hipExtent extent,
                                 size_t sizeBytes);

hipError_t ihipLaunchKernelCommand(amd::Command*& command, hipFunction_t f,
                                   uint32_t globalWorkSizeX, uint32_t globalWorkSizeY,
                                   uint32_t globalWorkSizeZ, uint32_t blockDimX, uint32_t blockDimY,
                                   uint32_t blockDimZ, uint32_t sharedMemBytes,
                                   amd::HostQueue* queue, void** kernelParams, void** extra,
                                   hipEvent_t startEvent, hipEvent_t stopEvent, uint32_t flags,
                                   uint32_t params, uint32_t gridId, uint32_t numGrids,
                                   uint64_t prevGridSum, uint64_t allGridSum, uint32_t firstDevice);

hipError_t ihipMemcpy3DCommand(amd::Command*& command, const hipMemcpy3DParms* p,
                               amd::HostQueue* queue);

hipError_t ihipMemsetCommand(std::vector<amd::Command*>& commands, void* dst, int64_t value,
                             size_t valueSize, size_t sizeBytes, amd::HostQueue* queue);

hipError_t ihipMemset3DCommand(std::vector<amd::Command*>& commands, hipPitchedPtr pitchedDevPtr,
                               int value, hipExtent extent, amd::HostQueue* queue);
