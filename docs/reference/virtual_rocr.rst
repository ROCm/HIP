.. meta::
  :description: This chapter lists user-mode API interfaces and libraries 
                necessary for host applications to launch compute kernels to 
                available HSA ROCm kernel agents.
  :keywords: AMD, ROCm, HIP, HSA, ROCR runtime, virtual memory management

*******************************************************************************
HSA runtime API for ROCm
*******************************************************************************

The following functions are located in the https://github.com/ROCm/ROCR-Runtime repository.

.. doxygenfunction:: hsa_amd_vmem_address_reserve

.. doxygenfunction:: hsa_amd_vmem_address_free

.. doxygenfunction:: hsa_amd_vmem_handle_create

.. doxygenfunction:: hsa_amd_vmem_handle_release

.. doxygenfunction:: hsa_amd_vmem_map

.. doxygenfunction:: hsa_amd_vmem_unmap

.. doxygenfunction:: hsa_amd_vmem_set_access

.. doxygenfunction:: hsa_amd_vmem_get_access

.. doxygenfunction:: hsa_amd_vmem_export_shareable_handle

.. doxygenfunction:: hsa_amd_vmem_import_shareable_handle

.. doxygenfunction:: hsa_amd_vmem_retain_alloc_handle

.. doxygenfunction:: hsa_amd_vmem_get_alloc_properties_from_handle
