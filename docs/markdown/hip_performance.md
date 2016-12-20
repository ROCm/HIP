# HIP Performance Optimizations

Please note that this document lists possible ways for experimenting with HIP stack to gain performance. Performance may vary from platform to platform.

### Unpinned Memory Transfer Optimizations
 
#### On Small BAR Setup

There are two possible ways to transfer data from host-to-device (H2D) and device-to-host(D2H)
 * Using Staging Buffers
 * Using PinInPlace

#### On Large BAR Setup

There are three possible ways to transfer data from host-to-device (H2D)
 * Using Staging Buffers
 * Using PinInPlace
 * Direct Memcpy
 
 And there are two possible ways to transfer data from device-to-host (D2H)
 * Using Staging Buffers
 * Using PinInPlace
 
Some GPUs may not be able to directly access host memory, and in these cases we need to
stage the copy through an optimized pinned staging buffer, to implement H2D and D2H copies.The copy is broken into buffer-sized chunks to limit the size of the buffer and also to provide better performance by overlapping the CPU copies with the DMA copies.

PinInPlace is another algorithm which pins the host memory "in-place", and copies it with the DMA engine.  

By default staging buffers are used for unpinned memory transfers. Environment variables allow control over the unpinned copy algorithm and parameters:

- HIP_PININPLACE - This environment variable forces the use of PinInPlace logic for all unpinned memory copies

- HIP_OPTIMAL_MEM_TRANSFER- This environment variable enables a hybrid memory copy logic based on thresholds. These thresholds can be managed with following environment variables:
  -   HIP_H2D_MEM_TRANSFER_THRESHOLD_STAGING_OR_PININPLACE - Threshold in bytes for H2D copy. For sizes smaller than threshold staging buffers logic would be used else PinInPlace logic.
  -   HIP_H2D_MEM_TRANSFER_THRESHOLD_DIRECT_OR_STAGING - Threshold in bytes for H2D copy. For sizes smaller than threshold direct copy logic would be used else staging buffers logic.
  -   HIP_D2H_MEM_TRANSFER_THRESHOLD - Threshold in bytes for D2H copy. For sizes smaller than threshold staging buffer logic would be used else PinInPlace logic.



