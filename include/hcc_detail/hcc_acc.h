#ifndef HCC_ACC_H
#define HCC_ACC_H
#include "hip/hip_runtime_api.h"

#if __cplusplus
#ifdef __HCC__
#include <hc.hpp>
/**
 * @brief Return hc::accelerator associated with the specified deviceId
 */
hipError_t hipHccGetAccelerator(int deviceId, hc::accelerator *acc);

/**
 * @brief Return hc::accelerator_view associated with the specified stream
 */
hipError_t hipHccGetAcceleratorView(hipStream_t stream, hc::accelerator_view **av);
#endif
#endif

#endif
