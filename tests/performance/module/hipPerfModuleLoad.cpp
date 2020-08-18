/*
Copyright (c) 2015-2017 Advanced Micro Devices, Inc. All rights reserved.
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

/* HIT_START
 * BUILD_CMD: hipPerfModuleLoad %hc -I%S/../../src %S/%s %S/../../src/test_common.cpp -o %T/%t EXCLUDE_HIP_PLATFORM nvcc
 * TEST: %t
 * HIT_END
 */

#include "test_common.h"

#include <vector>
#include <unordered_map>
#include <fstream>
#include <unistd.h>
#include <chrono>

#ifdef __unix__

#include <dirent.h>

//List of Download files
std::unordered_map<std::string, bool> TL_contents {
  {"Kernels.so", true},
  {"TensileLibrary.yaml", true},
  {"TensileLibrary_gfx803.co", true},
  {"TensileLibrary_gfx900.co", true},
  {"TensileLibrary_gfx906.co", true},
  {"TensileLibrary_gfx908.co", true},
  {"kernel_names.txt", true}
};

bool GetDirectoryContents(std::unordered_map<std::string, bool>& dir_contents) {
  DIR* dir = nullptr;
  struct dirent* ent = nullptr;

  //Open Current Directory
  if ((dir = opendir(".")) == 0) {
    std::cout<<"Failed to open current working directory, check permissions"<<std::endl;
    return false;
  }

  // Read the contents of the directory
  while ((ent = readdir(dir)) != nullptr) {
    dir_contents.insert({ent->d_name, true});
  }

  closedir(dir);
  return true;
}

bool ContentsAvailable() {
  std::unordered_map<std::string, bool> dir_contents;

  //Get recent directory contents
  if(!GetDirectoryContents(dir_contents)) {
    std::cout<<"Failed to get directory Contents"<<std::endl;
    return false;
  }

  //If the Tensile Library content is not present, then fail
  for (auto& TL_elem : TL_contents) {
    if (dir_contents.end() == dir_contents.find(TL_elem.first)) {
      std::cout<<"Failed to find the Tensile Library file: "<<TL_elem.first<<std::endl;
      return false;
    }
  }
  return true;
}

bool DownloadContents() {
  //Download the contents from TC repo
  std::cout<<"Downloading conents .... "<<std::endl;
  std::string wget_str = "wget -nH -q -N -r -np -R \" index.html* \" --cut-dirs=3 ";
  wget_str += "http://ocltc-backup.amd.com/hiptest/TensileLibrary/";
  system(wget_str.c_str());

  return true;
}

bool PreProcessContents() {
  // If Contents already available no other action needed
  if (ContentsAvailable()) {
    std::cout<<"Contents already available"<<std::endl;
    return true;
  }

  // Download the TL(Tensile Library) contents from TC
  if (!DownloadContents()) {
    std::cout<<"Failed to download contents"<<std::endl;
    return false;
  }

  //Check if downloaded contents are available
  if (!ContentsAvailable()) {
    std::cout<<"Failed to find TL contents even after download in CWD"<<std::endl;
    return false;
  }

  return true;
}

//Get Tensile Library File name, changes wrt target
bool getTLFileName(int device_id, std::string& tlf_name) {
  hipDeviceProp_t props;
  HIPCHECK(hipGetDeviceProperties(&props, device_id));
  std::string archName = props.gcnArchName;
  if (archName.size() <= 3) {
    std::cout<<"ArchName too small, Exiting"<<std::endl;
    HIPASSERT(false);
  }
  archName = archName.substr(3, (archName.size()-1));

  tlf_name = "TensileLibrary_gfx" + archName;
  tlf_name += ".co";
  return true;
}

bool RunTest(int device_id) {

  std::cout<<"For Device: "<<device_id<<std::endl;

  //Get Tensile Library File name, changes wrt target
  std::string tlf_name;
  if (!getTLFileName(device_id, tlf_name)) {
    return false;
  }

  //Measure Time taken for hipModuleLoad
  hipModule_t Module;
  auto mload_clock_start = std::chrono::steady_clock::now();
  HIPCHECK(hipModuleLoad(&Module, tlf_name.c_str()));
  auto mload_clock_stop = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::nano> mload_duration = (mload_clock_stop - mload_clock_start);
  std::cout<<"Time taken for hipModuleLoad : " <<std::chrono::duration_cast<std::chrono::nanoseconds>
    (mload_duration).count()<<" nanoseconds "<<std::endl;

  //Read kernels from a pre-populated text file
  std::string kernel_file_name = "kernel_names.txt";
  std::ifstream kernel_file(kernel_file_name);
  if (!kernel_file.is_open()) {
     std::cout<<"Failed to open Kernel File: "<<kernel_file_name<<std::endl;
     return false;
  }

  std::string kernel_line;
  std::vector<std::string> kernel_vec;
  while (std::getline(kernel_file, kernel_line)) {
    kernel_line.erase(std::remove(kernel_line.begin(), kernel_line.end(), '\r'),
                      kernel_line.end());
    kernel_vec.push_back(kernel_line);
  }

  //Measure the first hipModuleGetFunction
  hipFunction_t hfunc = nullptr;
  auto mgetf_clock_start = std::chrono::steady_clock::now();
  HIPCHECK(hipModuleGetFunction(&hfunc, Module, kernel_vec[0].c_str()));
  auto mgetf_clock_stop = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::nano> mgetf_duration = (mgetf_clock_stop - mgetf_clock_start);
  std::cout<<"Time taken to fetch a function via hipModuleGetFunction : "
    <<std::chrono::duration_cast<std::chrono::nanoseconds>
    (mgetf_duration).count()<<" nanoseconds "<<std::endl;

  //Measure the second hipModuleGetFunction
  hfunc = nullptr;
  mgetf_clock_start = std::chrono::steady_clock::now();
  HIPCHECK(hipModuleGetFunction(&hfunc, Module, kernel_vec[0].c_str()));
  mgetf_clock_stop = std::chrono::steady_clock::now();
  mgetf_duration = (mgetf_clock_stop - mgetf_clock_start);
  std::cout<<"Time taken fetch the same function via hipModuleGetFunction : "
    <<std::chrono::duration_cast<std::chrono::nanoseconds>
    (mgetf_duration).count()<<" nanoseconds "<<std::endl;

  double all_duration = 0;
  for (auto& kernel : kernel_vec) {
    hfunc = nullptr;
    mgetf_clock_start = std::chrono::steady_clock::now();
    HIPCHECK(hipModuleGetFunction(&hfunc, Module, kernel.c_str()));
    mgetf_clock_stop = std::chrono::steady_clock::now();
    mgetf_duration = (mgetf_clock_stop - mgetf_clock_start);
    all_duration += mgetf_duration.count();
  }

  if (kernel_vec.size() > 0) {
    std::cout << "Time taken for Average hipModuleGetFunction : "
      << (static_cast<double>(all_duration) / static_cast<double>(kernel_vec.size()))<<" nanoseconds "<<std::endl;
  }

  std::cout<<std::endl<<std::endl;

  HIPCHECK(hipModuleUnload(Module));
  return true;
}
#endif //__unix__

int main() {
  bool test_passed = true;

  do {
#ifdef __unix__
    //Preprocess contents for the test
    if (!PreProcessContents()) {
      std::cout<<"Failed in PreProcessContents step"<<std::endl;
      test_passed = false;
      break;
    }

    //Run the test for all devices
    int num_devices = 0;
    HIPCHECK(hipGetDeviceCount(&num_devices));
    for (int dev_idx = 0; dev_idx < num_devices; ++dev_idx) {
      if (!RunTest(dev_idx)) {
        test_passed = false;
        break;
      }
    }
#else
    std::cout<<"Detected non-linux Os. Skipping the test"<<std::endl;
#endif // __unix__
  } while(0);

  if (test_passed) {
    passed();
  }

  return 0;
}

