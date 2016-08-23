#include<hip_runtime.h>
#include<hip_runtime_api.h>
#include<iostream>
#include<fstream>
#include<hsa/hsa.h>
#include<hsa/amd_hsa_kernel_code.h>
#include<hsa/hsa_ven_amd_loader.h>
#include<vector>

#define LEN 64
#define SIZE LEN<<2

typedef hsa_code_object_t hipmodule;
typedef uint64_t hipfunction;
typedef unsigned int hipDevicePtr;

hsa_region_t systemRegion;
hsa_region_t kernArgRegion;
hsa_agent_t gpuAgent;
hsa_queue_t *Queue;
hsa_signal_t signal;

hsa_status_t findGpu(hsa_agent_t agent, void *data){
  hsa_device_type_t device_type;
  hsa_status_t hsa_error_code = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);
  if(hsa_error_code != HSA_STATUS_SUCCESS){return hsa_error_code;}
  if(device_type == HSA_DEVICE_TYPE_GPU){
    gpuAgent = agent;
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t FindRegions(hsa_region_t region, void *data){
  hsa_region_segment_t segment_id;
  hsa_region_get_info(region, HSA_REGION_INFO_SEGMENT, &segment_id);

  if (segment_id != HSA_REGION_SEGMENT_GLOBAL) {
    return HSA_STATUS_SUCCESS;
  }

  hsa_region_global_flag_t flags;
  hsa_region_get_info(region, HSA_REGION_INFO_GLOBAL_FLAGS, &flags);

  if(flags & HSA_REGION_GLOBAL_FLAG_FINE_GRAINED){
    systemRegion = region;
  }

  if(flags & HSA_REGION_GLOBAL_FLAG_KERNARG){
    kernArgRegion = region;
  }
  return HSA_STATUS_SUCCESS;
}

hipError_t ihipModuleLoad(hipmodule *module, const char *fname){
  std::ifstream in(fname, std::ios::binary | std::ios::ate);
  hipError_t ret = hipSuccess;
  if(!in){
    std::cout<<"Couldn't read file "<<fname<<std::endl;
    ret = hipErrorInvalidValue;
  }else{
    size_t size = std::string::size_type(in.tellg());
    void *p = NULL;
    hsa_status_t status = hsa_memory_allocate(systemRegion, size, (void**)&p);
    assert(status == HSA_STATUS_SUCCESS);
    char *ptr = (char*)p;
    if(!ptr){
      std::cout<<"Error: failed to allocate memory for code object."<<std::endl;
    }
    in.seekg(0, std::ios::beg);
    std::copy(std::istreambuf_iterator<char>(in),
              std::istreambuf_iterator<char>(), ptr);
    status = hsa_code_object_deserialize(ptr, size, NULL, module);
    if (status != HSA_STATUS_SUCCESS) { std::cout<<"Failed to deserialize code object"<<std::endl; }
    std::cout<<"File Read!"<<std::endl;
  }
  return ret;
}

hipError_t ihipModuleGetFunction(hipfunction *func, hipmodule hmod, const char *name){
  hsa_status_t status;
  hsa_executable_symbol_t kernel_symbol;
  hsa_executable_t executable;
  status = hsa_executable_create(HSA_PROFILE_FULL, HSA_EXECUTABLE_STATE_UNFROZEN, NULL, &executable);
  assert(status == HSA_STATUS_SUCCESS);
  status = hsa_executable_load_code_object(executable, gpuAgent, hmod, NULL);
  assert(status == HSA_STATUS_SUCCESS);
  status = hsa_executable_freeze(executable, NULL);
  assert(status == HSA_STATUS_SUCCESS);
  status = hsa_executable_get_symbol(executable, NULL, name, gpuAgent, 0, &kernel_symbol);
  assert(status == HSA_STATUS_SUCCESS);
  status = hsa_executable_symbol_get_info(kernel_symbol,
                    HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                    func);
  assert(status == HSA_STATUS_SUCCESS);
  return hipSuccess;
}

void hsaInit(){
  hsa_status_t status;
  uint32_t queue_size;
  status = hsa_iterate_agents(findGpu, NULL);
  assert(status == HSA_STATUS_SUCCESS);
  status = hsa_agent_iterate_regions(gpuAgent, FindRegions, NULL);
  assert(status == HSA_STATUS_SUCCESS);
  char agent_name[64];
  status = hsa_agent_get_info(gpuAgent, HSA_AGENT_INFO_NAME, agent_name);
  assert(status == HSA_STATUS_SUCCESS);
  status = hsa_agent_get_info(gpuAgent, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_size);
  assert(status == HSA_STATUS_SUCCESS);
  status = hsa_queue_create(gpuAgent, queue_size, HSA_QUEUE_TYPE_MULTI, NULL, NULL, UINT32_MAX, UINT32_MAX, &Queue);
  assert(status == HSA_STATUS_SUCCESS);
  status = hsa_signal_create(1, 0, NULL, &signal);
  assert(status == HSA_STATUS_SUCCESS);
  std::cout<<agent_name<<std::endl;
}

hipError_t hipDrvLaunchKernel(hipfunction f,
  uint32_t gridDimX, uint32_t gridDimY, uint32_t gridDimZ,
  uint32_t blockDimX, uint32_t blockDimY, uint32_t blockDimZ,
  uint32_t sharedMemBytes, hipStream_t hStream,
  void **kernelParams, uint32_t size, void **extra){
    void *kernarg;
    std::vector<void*>Args;
    void ***newP = (void***)kernelParams;
    for(uint32_t i=0;i<size/sizeof(void*);i++){
      Args.push_back((void*)*newP[i]);
    }

    hsa_status_t status = hsa_memory_allocate(kernArgRegion, size, &kernarg);
    memcpy(kernarg, Args.data(), size);
    const uint32_t queue_mask = Queue->size -1;
    uint32_t packet_index = hsa_queue_load_write_index_relaxed(Queue);
    hsa_kernel_dispatch_packet_t *dispatch_packet = &(((hsa_kernel_dispatch_packet_t*)(Queue->base_address))[packet_index & queue_mask]);
    dispatch_packet->completion_signal = signal;
    dispatch_packet->workgroup_size_x = blockDimX;
    dispatch_packet->workgroup_size_y = blockDimY;
    dispatch_packet->workgroup_size_z = blockDimZ;
    dispatch_packet->grid_size_x = blockDimX * gridDimX;
    dispatch_packet->grid_size_y = blockDimY * gridDimY;
    dispatch_packet->grid_size_z = blockDimZ * gridDimZ;
    dispatch_packet->group_segment_size = 0;
    dispatch_packet->private_segment_size = sharedMemBytes;
    dispatch_packet->kernarg_address = kernarg;
    dispatch_packet->kernel_object = (uint64_t)f;
    uint16_t header = (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
    (1 << HSA_PACKET_HEADER_BARRIER) |
    (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
    (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);
    uint16_t setup = 1 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
    uint32_t header32 = header | (setup << 16);
    __atomic_store_n((uint32_t*)(dispatch_packet), header32, __ATOMIC_RELEASE);
    hsa_queue_store_write_index_relaxed(Queue, packet_index+1);
    hsa_signal_store_relaxed(Queue->doorbell_signal, packet_index);
    hsa_signal_value_t value = hsa_signal_wait_acquire(signal, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
    return hipSuccess;
}

#define fileName "vcpy_isa.co"
#define kernel_name "hello_world"

__global__ void Cpy(hipLaunchParm lp, float *Ad, float* Bd){
  int tx = hipThreadIdx_x;
  Bd[tx] = Ad[tx];
}

amd_kernel_code_t* getAkc(uint64_t handle){
    bool ext_supported = false;
    hsa_status_t status = hsa_system_extension_supported(
                HSA_EXTENSION_AMD_LOADER, 1, 0, &ext_supported);
    assert(HSA_STATUS_SUCCESS == status);
    assert(true == ext_supported);
    hsa_ven_amd_loader_1_00_pfn_t ext_table = {nullptr};
    status = hsa_system_get_extension_table(
        HSA_EXTENSION_AMD_LOADER, 1, 0, &ext_table);
    assert(HSA_STATUS_SUCCESS == status);
    assert(nullptr != ext_table.hsa_ven_amd_loader_query_host_address);
    std::cout<<"Start"<<std::endl;
    const void *akc = nullptr;//new amd_kernel_code_t;
    std::cout<<"End"<<std::endl;
    status = ext_table.hsa_ven_amd_loader_query_host_address(
      reinterpret_cast<void*>(handle), &akc);

    if(HSA_STATUS_SUCCESS != status){
        akc = reinterpret_cast<void*>(handle);
    }

    assert(nullptr!=akc);
    amd_kernel_code_t *Akc = (amd_kernel_code_t*)akc;
    std::cout<<Akc->kernarg_segment_byte_size<<std::endl;
    return nullptr;
}

int main(){
  float *A, *B, *Ad, *Bd;
  hipDevicePtr Aptr, Bptr;
  A = new float[LEN];
  B = new float[LEN];

  for(uint32_t i=0;i<LEN;i++){
    A[i] = i*1.0f;
    B[i] = 0.0f;
    std::cout<<A[i] << " "<<B[i]<<std::endl;
  }

  hipMalloc((void**)&Ad, SIZE);
  hipMalloc((void**)&Bd, SIZE);
  Aptr = reinterpret_cast<unsigned long>(Ad);
  Bptr = reinterpret_cast<unsigned long>(Bd);
  hsaInit();

  hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice);
  hipMemcpy(Bd, B, SIZE, hipMemcpyHostToDevice);
  hipModule Module;
  hipFunction Function;
  hipModuleLoad(&Module, fileName);
  hipModuleGetFunction(&Function, Module, kernel_name);
  hipStream_t stream;
  hipStreamCreate(&stream);
  void *args[2] = {&Ad, &Bd};

/*  struct __attribute__((aligned(16))) args_t{
    void *Aptr;
    void *Bptr;
  } args;
  args.Aptr = Ad;
  args.Bptr = Bd;
*/
//  hipDrvLaunchKernel(Function, 1, 1, 1, LEN, 1, 1, 0, 0, (void**)&args, sizeof(args), 0);

    amd_kernel_code_t *akc = getAkc(Function.kernel);

    std::vector<void*>argBuffer(2);
    memcpy(&argBuffer[0], &Ad, sizeof(void*));
    memcpy(&argBuffer[1], &Bd, sizeof(void*));

    size_t size = argBuffer.size()*sizeof(void*);

    void *config[] = {
      HIP_LAUNCH_PARAM_BUFFER_POINTER, &argBuffer[0],
      HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
      HIP_LAUNCH_PARAM_END
    };

    hipLaunchModuleKernel(Function, 1, 1, 1, LEN, 1, 1, 0, stream, NULL, (void**)&config); 

    hipStreamDestroy(stream);

//  hipLaunchKernel(Cpy, dim3(1), dim3(LEN), 0, 0, Ad, Bd);
 hipMemcpy(B, Bd, SIZE, hipMemcpyDeviceToHost);

  for(uint32_t i=0;i<LEN;i++){
  std::cout<<A[i]<<" - "<<B[i]<<std::endl;
  }

  return 0;
}
