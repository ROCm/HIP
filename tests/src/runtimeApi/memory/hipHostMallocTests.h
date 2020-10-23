/*
Copyright (c) 2020-Present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef TESTS_SRC_RUNTIMEAPI_MEMORY_HIPHOSTMALLOCTESTS_H_
#define TESTS_SRC_RUNTIMEAPI_MEMORY_HIPHOSTMALLOCTESTS_H_

#include "test_common.h"

/**
 * @brief Error codes retured by rocm_smi_lib functions
 */
typedef enum {
  RSMI_STATUS_SUCCESS = 0x0,             //!< Operation was successful
  RSMI_STATUS_INVALID_ARGS,              //!< Passed in arguments are not valid
  RSMI_STATUS_NOT_SUPPORTED,             //!< The requested information or
                                         //!< action is not available for the
                                         //!< given input, on the given system
  RSMI_STATUS_FILE_ERROR,                //!< Problem accessing a file. This
                                         //!< may because the operation is not
                                         //!< supported by the Linux kernel
                                         //!< version running on the executing
                                         //!< machine
  RSMI_STATUS_PERMISSION,                //!< Permission denied/EACCESS file
                                         //!< error. Many functions require
                                         //!< root access to run.
  RSMI_STATUS_OUT_OF_RESOURCES,          //!< Unable to acquire memory or other
                                         //!< resource
  RSMI_STATUS_INTERNAL_EXCEPTION,        //!< An internal exception was caught
  RSMI_STATUS_INPUT_OUT_OF_BOUNDS,       //!< The provided input is out of
                                         //!< allowable or safe range
  RSMI_STATUS_INIT_ERROR,                //!< An error occurred when rsmi
                                         //!< initializing internal data
                                         //!< structures
  RSMI_INITIALIZATION_ERROR = RSMI_STATUS_INIT_ERROR,
  RSMI_STATUS_NOT_YET_IMPLEMENTED,       //!< The requested function has not
                                         //!< yet been implemented in the
                                         //!< current system for the current
                                         //!< devices
  RSMI_STATUS_NOT_FOUND,                 //!< An item was searched for but not
                                         //!< found
  RSMI_STATUS_INSUFFICIENT_SIZE,         //!< Not enough resources were
                                         //!< available for the operation
  RSMI_STATUS_INTERRUPT,                 //!< An interrupt occurred during
                                         //!< execution of function
  RSMI_STATUS_UNEXPECTED_SIZE,           //!< An unexpected amount of data
                                         //!< was read
  RSMI_STATUS_NO_DATA,                   //!< No data was found for a given
                                         //!< input
  RSMI_STATUS_UNEXPECTED_DATA,           //!< The data read or provided to
                                         //!< function is not what was expected
  RSMI_STATUS_BUSY,                      //!< A resource or mutex could not be
                                         //!< acquired because it is already
                                         //!< being used
  RSMI_STATUS_REFCOUNT_OVERFLOW,          //!< An internal reference counter
                                         //!< exceeded INT32_MAX

  RSMI_STATUS_UNKNOWN_ERROR = 0xFFFFFFFF,  //!< An unknown error occurred
} rsmi_status_t;


/**
 * @brief Types of memory
 */
typedef enum {
  RSMI_MEM_TYPE_FIRST = 0,

  RSMI_MEM_TYPE_VRAM = RSMI_MEM_TYPE_FIRST,  //!< VRAM memory
  RSMI_MEM_TYPE_VIS_VRAM,                    //!< VRAM memory that is visible
  RSMI_MEM_TYPE_GTT,                         //!< GTT memory

  RSMI_MEM_TYPE_LAST = RSMI_MEM_TYPE_GTT
} rsmi_memory_type_t;



#endif  // TESTS_SRC_RUNTIMEAPI_MEMORY_HIPHOSTMALLOCTESTS_H_
