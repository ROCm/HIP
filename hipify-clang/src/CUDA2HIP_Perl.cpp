/*
Copyright (c) 2015 - present Advanced Micro Devices, Inc. All rights reserved.

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

#include <sstream>
#include <regex>
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Path.h"
#include "CUDA2HIP.h"
#include "CUDA2HIP_Scripting.h"
#include "ArgParse.h"
#include "StringUtils.h"
#include "LLVMCompat.h"
#include "Statistics.h"

using namespace llvm;

namespace perl {

  const std::string sCopyright =
    "##\n"
    "# Copyright (c) 2015-present Advanced Micro Devices, Inc. All rights reserved.\n"
    "#\n"
    "# Permission is hereby granted, free of charge, to any person obtaining a copy\n"
    "# of this software and associated documentation files (the \"Software\"), to deal\n"
    "# in the Software without restriction, including without limitation the rights\n"
    "# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\n"
    "# copies of the Software, and to permit persons to whom the Software is\n"
    "# furnished to do so, subject to the following conditions:\n"
    "#\n"
    "# The above copyright notice and this permission notice shall be included in\n"
    "# all copies or substantial portions of the Software.\n"
    "#\n"
    "# THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\n"
    "# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\n"
    "# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE\n"
    "# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\n"
    "# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\n"
    "# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN\n"
    "# THE SOFTWARE.\n"
    "##\n";

  const std::string tab = "    ";
  const std::string double_tab = tab + tab;
  const std::string triple_tab = double_tab + tab;
  const std::string quad_tab = triple_tab + tab;
  const std::string tab_5 = quad_tab + tab;
  const std::string tab_6 = tab_5 + tab;
  const std::string sSub = "sub";
  const std::string sReturn_0 = "return 0;\n";
  const std::string sReturn_k = "return $k;\n";
  const std::string sForeach = "foreach $func (\n";
  const std::string sMy = "my ";
  const std::string sMy_k = sMy + "$k = 0;";
  const std::string sNoWarns = "no warnings qw/uninitialized/;";

  const std::string sCudaDevice = "cudaDevice";
  const std::string sCudaDeviceId = "cudaDeviceId";
  const std::string sCudaDevices = "cudaDevices";
  const std::string sCudaDevice_t = "cudaDevice_t";
  const std::string sCudaIDs = "cudaIDs";
  const std::string sCudaGridDim = "cudaGridDim";
  const std::string sCudaDimGrid = "cudaDimGrid";
  const std::string sCudaDimBlock = "cudaDimBlock";
  const std::string sCudaGradInput = "cudaGradInput";
  const std::string sCudaGradOutput = "cudaGradOutput";
  const std::string sCudaInput = "cudaInput";
  const std::string sCudaOutput = "cudaOutput";
  const std::string sCudaIndices = "cudaIndices";
  const std::string sCudaGaugeField = "cudaGaugeField";
  const std::string sCudaMom = "cudaMom";
  const std::string sCudaGauge = "cudaGauge";
  const std::string sCudaInGauge = "cudaInGauge";
  const std::string sCudaColorSpinorField = "cudaColorSpinorField";
  const std::string sCudaSiteLink = "cudaSiteLink";
  const std::string sCudaFatLink = "cudaFatLink";
  const std::string sCudaStaple = "cudaStaple";
  const std::string sCudaCloverField = "cudaCloverField";
  const std::string sCudaParam = "cudaParam";

  const std::set<std::string> Whitelist{
    {sCudaDevice}, {sCudaDevice_t}, {sCudaIDs}, {sCudaGridDim}, {sCudaDimGrid}, {sCudaDimBlock}, {sCudaDeviceId}, {sCudaDevices},
    {sCudaGradInput}, {sCudaGradOutput}, {sCudaInput}, {sCudaOutput}, {sCudaIndices}, {sCudaGaugeField}, {sCudaMom}, {sCudaGauge},
    {sCudaInGauge}, {sCudaColorSpinorField}, {sCudaSiteLink}, {sCudaFatLink}, {sCudaStaple}, {sCudaCloverField}, {sCudaParam}
  };

  void generateHeader(std::unique_ptr<std::ostream>& streamPtr) {
    *streamPtr.get() << "#!/usr/bin/perl -w" << std::endl << std::endl;
    *streamPtr.get() << sCopyright << std::endl;
    *streamPtr.get() << "#usage hipify-perl [OPTIONS] INPUT_FILE" << std::endl << std::endl;
    *streamPtr.get() << "use Getopt::Long;" << std::endl;
    *streamPtr.get() << sMy << "$whitelist = \"\";" << std::endl;
    *streamPtr.get() << sMy << "$fileName = \"\";" << std::endl;
    *streamPtr.get() << sMy << "%ft;" << std::endl;
    *streamPtr.get() << sMy << "%Tkernels;" << std::endl << std::endl;
    *streamPtr.get() << "GetOptions(" << std::endl;
    *streamPtr.get() << tab << "  \"examine\" => \\$examine                  # Combines -no-output and -print-stats options." << std::endl;
    *streamPtr.get() << tab << ", \"inplace\" => \\$inplace                  # Modify input file inplace, replacing input with hipified output, save backup in .prehip file." << std::endl;
    *streamPtr.get() << tab << ", \"no-output\" => \\$no_output              # Don't write any translated output to stdout." << std::endl;
    *streamPtr.get() << tab << ", \"print-stats\" => \\$print_stats          # Print translation statistics." << std::endl;
    *streamPtr.get() << tab << ", \"quiet-warnings\" => \\$quiet_warnings    # Don't print warnings on unknown CUDA functions." << std::endl;
    *streamPtr.get() << tab << ", \"whitelist=s\" => \\$whitelist            # TODO: test it beforehand" << std::endl;
    *streamPtr.get() << ");" << std::endl << std::endl;
    *streamPtr.get() << "$print_stats = 1 if $examine;" << std::endl;
    *streamPtr.get() << "$no_output = 1 if $examine;" << std::endl << std::endl;
    *streamPtr.get() << "# Whitelist of cuda[A-Z] identifiers, which are commonly used in CUDA sources but don't map to any CUDA API:" << std::endl;
    *streamPtr.get() << "@whitelist = (";
    unsigned int num = 0;
    for (const std::string &m : Whitelist) {
      *streamPtr.get() << std::endl << tab << (num ? ", " : "  ") << "\"" << m << "\"";
      ++num;
    }
    *streamPtr.get() << std::endl << ");" << std::endl << std::endl;
    *streamPtr.get() << "push(@whitelist, split(',', $whitelist));" << std::endl << std::endl;
  }

  void generateStatFunctions(std::unique_ptr<std::ostream>& streamPtr) {
    *streamPtr.get() << std::endl << sSub << " totalStats" << " {" << std::endl;
    *streamPtr.get() << tab << sMy << "%count = %{ shift() };" << std::endl;
    *streamPtr.get() << tab << sMy << "$total = 0;" << std::endl;
    *streamPtr.get() << tab << "foreach $key (keys %count) {" << std::endl;
    *streamPtr.get() << double_tab << "$total += $count{$key};" << std::endl << tab << "}" << std::endl;
    *streamPtr.get() << tab << "return $total;" << std::endl << "};" << std::endl;
    *streamPtr.get() << std::endl << sSub << " printStats" << " {" << std::endl;
    *streamPtr.get() << tab << sMy << "$label     = shift();" << std::endl;
    *streamPtr.get() << tab << sMy << "@statNames = @{ shift() };" << std::endl;
    *streamPtr.get() << tab << sMy << "%counts    = %{ shift() };" << std::endl;
    *streamPtr.get() << tab << sMy << "$warnings  = shift();" << std::endl;
    *streamPtr.get() << tab << sMy << "$loc       = shift();" << std::endl;
    *streamPtr.get() << tab << sMy << "$total     = totalStats(\\%counts);" << std::endl;
    *streamPtr.get() << tab << "printf STDERR \"%s %d CUDA->HIP refs ( \", $label, $total;" << std::endl;
    *streamPtr.get() << tab << "foreach $stat (@statNames) {" << std::endl;
    *streamPtr.get() << double_tab << "printf STDERR \"%s:%d \", $stat, $counts{$stat};" << std::endl;
    *streamPtr.get() << tab << "}" << std::endl;
    *streamPtr.get() << tab << "printf STDERR \")\\n  warn:%d LOC:%d\", $warnings, $loc;" << std::endl << "}" << std::endl;
    for (int i = 0; i < 2; ++i) {
      *streamPtr.get() << std::endl << sSub << " " << (i ? "clearStats" : "addStats") << " {" << std::endl;
      *streamPtr.get() << tab << sMy << "$dest_ref  = shift();" << std::endl;
      *streamPtr.get() << tab << sMy << (i ? "@statNames = @{ shift() };" : "%adder     = %{ shift() };") << std::endl;
      *streamPtr.get() << tab << "foreach " << (i ? "$stat(@statNames)" : "$key (keys %adder)") << " {" << std::endl;
      *streamPtr.get() << double_tab << "$dest_ref->" << (i ? "{$stat} = 0;" : "{$key} += $adder{$key};") << std::endl << tab << "}" << std::endl << "}" << std::endl;
    }
  }

  void generateSimpleSubstitutions(std::unique_ptr<std::ostream>& streamPtr) {
    *streamPtr.get() << std::endl << sSub << " simpleSubstitutions" << " {" << std::endl;
    for (int i = 0; i < NUM_CONV_TYPES; ++i) {
      if (i == CONV_INCLUDE_CUDA_MAIN_H || i == CONV_INCLUDE) {
        for (auto& ma : CUDA_INCLUDE_MAP) {
          if (Statistics::isUnsupported(ma.second)) continue;
          if (i == ma.second.type) {
            std::string sCUDA = ma.first.str();
            std::string sHIP = ma.second.hipName.str();
            sCUDA = std::regex_replace(sCUDA, std::regex("/"), "\\/");
            sHIP = std::regex_replace(sHIP, std::regex("/"), "\\/");
            *streamPtr.get() << tab << "$ft{'" << counterNames[ma.second.type] << "'} += s/\\b" << sCUDA << "\\b/" << sHIP << "/g;" << std::endl;
          }
        }
      } else {
        for (auto& ma : CUDA_RENAMES_MAP()) {
          if (Statistics::isUnsupported(ma.second)) continue;
          if (i == ma.second.type) {
            *streamPtr.get() << tab << "$ft{'" << counterNames[ma.second.type] << "'} += s/\\b" << ma.first.str() << "\\b/" << ma.second.hipName.str() << "/g;" << std::endl;
          }
        }
      }
    }
    *streamPtr.get() << "}" << std::endl;
  }

  void generateExternShared(std::unique_ptr<std::ostream>& streamPtr) {
    *streamPtr.get() << std::endl << "# CUDA extern __shared__ syntax replace with HIP_DYNAMIC_SHARED() macro" << std::endl;
    *streamPtr.get() << sSub << " transformExternShared" << " {" << std::endl;
    *streamPtr.get() << tab << sNoWarns << std::endl;
    *streamPtr.get() << tab << sMy_k << std::endl;
    *streamPtr.get() << tab << "$k += s/extern\\s+([\\w\\(\\)]+)?\\s*__shared__\\s+([\\w:<>\\s]+)\\s+(\\w+)\\s*\\[\\s*\\]\\s*;/HIP_DYNAMIC_SHARED($1 $2, $3)/g;" << std::endl;
    *streamPtr.get() << tab << "$ft{'extern_shared'} += $k;" << std::endl << "}" << std::endl;
  }

  void generateKernelLaunch(std::unique_ptr<std::ostream>& streamPtr) {
    *streamPtr.get() << std::endl << "# CUDA Kernel Launch Syntax" << std::endl;
    *streamPtr.get() << sSub << " transformKernelLaunch" << " {" << std::endl;
    *streamPtr.get() << tab << sNoWarns << std::endl;
    *streamPtr.get() << tab << sMy_k << std::endl << std::endl;

    *streamPtr.get() << tab << "# Handle the kern<...><<<Dg, Db, Ns, S>>>() syntax with empty args:" << std::endl;
    *streamPtr.get() << tab << "$k += s/(\\w+)\\s*<(.+)>\\s*<<<\\s*(.+)\\s*,\\s*(.+)\\s*,\\s*(.+)\\s*,\\s*(.+)\\s*>>>(\\s*)\\((\\s*)\\)/hipLaunchKernelGGL(HIP_KERNEL_NAME($1<$2>), dim3($3), dim3($4), $5, $6)/g;" << std::endl;
    *streamPtr.get() << tab << "# Handle the kern<<<Dg, Db, Ns, S>>>() syntax with empty args:" << std::endl;
    *streamPtr.get() << tab << "$k += s/(\\w+)\\s*<<<\\s*(.+)\\s*,\\s*(.+)\\s*,\\s*(.+)\\s*,\\s*(.+)\\s*>>>(\\s*)\\((\\s*)\\)/hipLaunchKernelGGL($1, dim3($2), dim3($3), $4, $5)/g;" << std::endl << std::endl;

    *streamPtr.get() << tab << "# Handle the kern<...><<<Dg, Db, Ns, S>>>(...) syntax with non-empty args:" << std::endl;
    *streamPtr.get() << tab << "$k += s/(\\w+)\\s*<(.+)>\\s*<<<\\s*(.+)\\s*,\\s*(.+)\\s*,\\s*(.+)\\s*,\\s*(.+)\\s*>>>(\\s*)\\(/hipLaunchKernelGGL(HIP_KERNEL_NAME($1<$2>), dim3($3), dim3($4), $5, $6, /g;" << std::endl;
    *streamPtr.get() << tab << "# Handle the kern<<<Dg, Db, Ns, S>>>(...) syntax with non-empty args:" << std::endl;
    *streamPtr.get() << tab << "$k += s/(\\w+)\\s*<<<\\s*(.+)\\s*,\\s*(.+)\\s*,\\s*(.+)\\s*,\\s*(.+)\\s*>>>(\\s*)\\(/hipLaunchKernelGGL($1, dim3($2), dim3($3), $4, $5, /g;" << std::endl << std::endl;

    *streamPtr.get() << tab << "# Handle the kern<...><<<Dg, Db, Ns>>>() syntax with empty args:" << std::endl;
    *streamPtr.get() << tab << "$k += s/(\\w+)\\s*<(.+)>\\s*<<<\\s*(.+)\\s*,\\s*(.+)\\s*,\\s*(.+)\\s*>>>(\\s*)\\((\\s*)\\)/hipLaunchKernelGGL(HIP_KERNEL_NAME($1<$2>), dim3($3), dim3($4), $5, 0)/g;" << std::endl;
    *streamPtr.get() << tab << "# Handle the kern<<<Dg, Db, Ns>>>() syntax with empty args:" << std::endl;
    *streamPtr.get() << tab << "$k += s/(\\w+)\\s*<<<\\s*(.+)\\s*,\\s*(.+)\\s*,\\s*(.+)\\s*>>>(\\s*)\\((\\s*)\\)/hipLaunchKernelGGL($1, dim3($2), dim3($3), $4, 0)/g;" << std::endl << std::endl;

    *streamPtr.get() << tab << "# Handle the kern<...><<Dg, Db, Ns>>>(...) syntax with non-empty args:" << std::endl;
    *streamPtr.get() << tab << "$k += s/(\\w+)\\s*<(.+)>\\s*<<<\\s*(.+)\\s*,\\s*(.+)\\s*,\\s*(.+)\\s*>>>(\\s*)\\(/hipLaunchKernelGGL(HIP_KERNEL_NAME($1<$2>), dim3($3), dim3($4), $5, 0, /g;" << std::endl;
    *streamPtr.get() << tab << "# Handle the kern<<<Dg, Db, Ns>>>(...) syntax with non-empty args:" << std::endl;
    *streamPtr.get() << tab << "$k += s/(\\w+)\\s*<<<\\s*(.+)\\s*,\\s*(.+)\\s*,\\s*(.+)\\s*>>>(\\s*)\\(/hipLaunchKernelGGL($1, dim3($2), dim3($3), $4, 0, /g;" << std::endl << std::endl;

    *streamPtr.get() << tab << "# Handle the kern<...><<<Dg, Db>>>() syntax with empty args:" << std::endl;
    *streamPtr.get() << tab << "$k += s/(\\w+)\\s*<(.+)>\\s*<<<\\s*(.+)\\s*,\\s*(.+)\\s*>>>(\\s*)\\((\\s*)\\)/hipLaunchKernelGGL(HIP_KERNEL_NAME($1<$2>), dim3($3), dim3($4), 0, 0)/g;" << std::endl;
    *streamPtr.get() << tab << "# Handle the kern<<<Dg, Db>>>() syntax with empty args:" << std::endl;
    *streamPtr.get() << tab << "$k += s/(\\w+)\\s*<<<\\s*(.+)\\s*,\\s*(.+)\\s*>>>(\\s*)\\((\\s*)\\)/hipLaunchKernelGGL($1, dim3($2), dim3($3), 0, 0)/g;" << std::endl << std::endl;

    *streamPtr.get() << tab << "# Handle the kern<...><<<Dg, Db>>>(...) syntax with non-empty args:" << std::endl;
    *streamPtr.get() << tab << "$k += s/(\\w+)\\s*<(.+)>\\s*<<<\\s*(.+)\\s*,\\s*(.+)\\s*>>>(\\s*)\\(/hipLaunchKernelGGL(HIP_KERNEL_NAME($1<$2>), dim3($3), dim3($4), 0, 0, /g;" << std::endl;
    *streamPtr.get() << tab << "# Handle the kern<<<Dg, Db>>>(...) syntax with non-empty args:" << std::endl;
    *streamPtr.get() << tab << "$k += s/(\\w+)\\s*<<<\\s*(.+)\\s*,\\s*(.+)\\s*>>>(\\s*)\\(/hipLaunchKernelGGL($1, dim3($2), dim3($3), 0, 0, /g;" << std::endl << std::endl;

    *streamPtr.get() << tab << "if ($k) {" << std::endl;
    *streamPtr.get() << double_tab << "$ft{'kernel_launch'} += $k;" << std::endl;
    *streamPtr.get() << double_tab << "$Tkernels{$1}++;" << std::endl << tab << "}" << std::endl << "}" << std::endl;
  }

  void generateHostFunctions(std::unique_ptr<std::ostream>& streamPtr) {
    *streamPtr.get() << std::endl << sSub << " transformHostFunctions" << " {" << std::endl << tab << sMy_k << std::endl;
    std::set<std::string> &funcSet = DeviceSymbolFunctions0;
    const std::string s0 = "$k += s/(?<!\\/\\/ CHECK: )($func)\\s*\\(\\s*([^,]+)\\s*,/$func\\(";
    const std::string s1 = "$k += s/(?<!\\/\\/ CHECK: )($func)\\s*\\(\\s*([^,]+)\\s*,\\s*([^,\\)]+)\\s*(,\\s*|\\))\\s*/$func\\($2, ";
    for (int i = 0; i < 4; ++i) {
      *streamPtr.get() << tab + sForeach;
      switch (i) {
      case 1: funcSet = DeviceSymbolFunctions1; break;
      case 2: funcSet = ReinterpretFunctions0; break;
      case 3: funcSet = ReinterpretFunctions1; break;
      default: funcSet = DeviceSymbolFunctions0;
      }
      unsigned int count = 0;
      for (auto& f : funcSet) {
        const auto found = CUDA_RUNTIME_FUNCTION_MAP.find(f);
        if (found != CUDA_RUNTIME_FUNCTION_MAP.end()) {
          *streamPtr.get() << (count ? ",\n" : "") << double_tab << "\"" << found->second.hipName.str() << "\"";
          count++;
        }
      }
      *streamPtr.get() << std::endl << tab << ")" << std::endl << tab << "{" << std::endl << double_tab;
      switch (i) {
      case 0:
      default:
        *streamPtr.get() << s0 << sHIP_SYMBOL << "\\($2\\),/g" << std::endl; break;
      case 1:
        *streamPtr.get() << s1 << sHIP_SYMBOL << "\\($3\\)$4/g;" << std::endl; break;
      case 2:
        *streamPtr.get() << s0 << s_reinterpret_cast << "\\($2\\),/g" << std::endl; break;
      case 3:
        *streamPtr.get() << s1 << s_reinterpret_cast << "\\($3\\)$4/g;" << std::endl; break;
      }
      *streamPtr.get() << tab << "}" << std::endl;
    }
    *streamPtr.get() << tab << sReturn_k << "}" << std::endl;
  }

  void generateDeviceFunctions(std::unique_ptr<std::ostream>& streamPtr) {
    unsigned int countUnsupported = 0;
    unsigned int countSupported = 0;
    std::stringstream sSupported;
    std::stringstream sUnsupported;
    for (auto& ma : CUDA_DEVICE_FUNC_MAP) {
      bool isUnsupported = Statistics::isUnsupported(ma.second);
      (isUnsupported ? sUnsupported : sSupported) << ((isUnsupported && countUnsupported) || (!isUnsupported && countSupported) ? ",\n" : "") << double_tab << "\"" << ma.first.str() << "\"";
      if (isUnsupported) countUnsupported++;
      else countSupported++;
    }
    std::stringstream subCountSupported;
    std::stringstream subWarnUnsupported;
    std::stringstream subCommon;
    std::string sCommon = tab + sMy_k + "\n" + tab + sForeach;
    subCountSupported << std::endl << sSub << " countSupportedDeviceFunctions" << " {" << std::endl << (countSupported ? sCommon : tab + sReturn_0);
    subWarnUnsupported << std::endl << sSub << " warnUnsupportedDeviceFunctions" << " {" << std::endl << (countUnsupported ? tab + sMy + "$line_num = shift;\n" + sCommon : tab + sReturn_0);
    if (countSupported) {
      subCountSupported << sSupported.str() << std::endl << tab << ")" << std::endl;
    }
    if (countUnsupported) {
      subWarnUnsupported << sUnsupported.str() << std::endl << tab << ")" << std::endl;
    }
    if (countSupported || countUnsupported) {
      subCommon << tab << "{" << std::endl;
      subCommon << double_tab << "# match device function from the list, except those, which have a namespace prefix (aka somenamespace::umin(...));" << std::endl;
      subCommon << double_tab << "# function with only global namespace qualifier '::' (aka ::umin(...)) should be treated as a device function (and warned as well as without such qualifier);" << std::endl;
      subCommon << double_tab << sMy << "$mt_namespace = m/(\\w+)::($func)\\s*\\(\\s*.*\\s*\\)/g;" << std::endl;
      subCommon << double_tab << sMy << "$mt = m/($func)\\s*\\(\\s*.*\\s*\\)/g;" << std::endl;
      subCommon << double_tab << "if ($mt && !$mt_namespace) {" << std::endl;
      subCommon << triple_tab << "$k += $mt;" << std::endl;
    }
    if (countSupported) {
      subCountSupported << subCommon.str();
    }
    if (countUnsupported) {
      subWarnUnsupported << subCommon.str();
      subWarnUnsupported << triple_tab << "print STDERR \"  warning: $fileName:$line_num: unsupported device function \\\"$func\\\": $_\\n\";" << std::endl;
    }
    if (countSupported || countUnsupported) {
      sCommon = double_tab + "}\n" + tab + "}\n" + tab + sReturn_k;
    }
    if (countSupported) subCountSupported << sCommon;
    if (countUnsupported) subWarnUnsupported << sCommon;
    subCountSupported << "}" << std::endl;
    subWarnUnsupported << "}" << std::endl;
    *streamPtr.get() << subCountSupported.str();
    *streamPtr.get() << subWarnUnsupported.str();
  }

  bool generate(bool Generate) {
    if (!Generate) return true;
    std::string dstHipifyPerl = "hipify-perl", dstHipifyPerlDir = OutputHipifyPerlDir;
    std::error_code EC;
    if (!dstHipifyPerlDir.empty()) {
      std::string sOutputHipifyPerlDirAbsPath = getAbsoluteDirectoryPath(OutputHipifyPerlDir, EC, "output hipify-perl");
      if (EC) return false;
      dstHipifyPerl = sOutputHipifyPerlDirAbsPath + "/" + dstHipifyPerl;
    }
    SmallString<128> tmpFile;
    StringRef ext = "hipify-perl-tmp";
    EC = sys::fs::createTemporaryFile(dstHipifyPerl, ext, tmpFile);
    if (EC) {
      llvm::errs() << "\n" << sHipify << sError << EC.message() << ": " << tmpFile << "\n";
      return false;
    }
    std::unique_ptr<std::ostream> streamPtr = std::unique_ptr<std::ostream>(new std::ofstream(tmpFile.c_str(), std::ios_base::trunc));
    generateHeader(streamPtr);
    std::string sConv = sMy + "$apiCalls   = ";
    unsigned int exclude[3] = { CONV_DEVICE_FUNC, CONV_EXTERN_SHARED, CONV_KERNEL_LAUNCH };
    *streamPtr.get() << "@statNames = (";
    for (unsigned int i = 0; i < NUM_CONV_TYPES - 1; ++i) {
      *streamPtr.get() << "\"" << counterNames[i] << "\", ";
      if (std::any_of(exclude, exclude + 3, [&i](unsigned int x) { return x == i; })) continue;
      sConv += "$ft{'" + std::string(counterNames[i]) + "'}" + (i < NUM_CONV_TYPES - 2 ? " + " : ";");
    }
    if (sConv.back() == ' ') {
      sConv = sConv.substr(0, sConv.size() - 3) + ";";
    }
    *streamPtr.get() << "\"" << counterNames[NUM_CONV_TYPES - 1] << "\");" << std::endl;
    generateStatFunctions(streamPtr);
    generateSimpleSubstitutions(streamPtr);
    generateExternShared(streamPtr);
    generateKernelLaunch(streamPtr);
    generateHostFunctions(streamPtr);
    generateDeviceFunctions(streamPtr);
    *streamPtr.get() << std::endl << "# Count of transforms in all files" << std::endl;
    *streamPtr.get() << sMy << "%tt;" << std::endl;
    *streamPtr.get() << "clearStats(\\%tt, \\@statNames);" << std::endl;
    *streamPtr.get() << "$Twarnings = 0;" << std::endl;
    *streamPtr.get() << "$TlineCount = 0;" << std::endl;
    *streamPtr.get() << sMy << "%TwarningTags;" << std::endl;
    *streamPtr.get() << sMy << "$fileCount = @ARGV;" << std::endl << std::endl;
    *streamPtr.get() << "while (@ARGV) {" << std::endl;
    *streamPtr.get() << tab << "$fileName=shift (@ARGV);" << std::endl;
    *streamPtr.get() << tab << "if ($inplace) {" << std::endl;
    *streamPtr.get() << double_tab << sMy << "$file_prehip = \"$fileName\" . \".prehip\";" << std::endl;
    *streamPtr.get() << double_tab << sMy << "$infile;" << std::endl;
    *streamPtr.get() << double_tab << sMy << "$outfile;" << std::endl;
    *streamPtr.get() << double_tab << "if (-e $file_prehip) {" << std::endl;
    *streamPtr.get() << triple_tab << "$infile  = $file_prehip;" << std::endl;
    *streamPtr.get() << triple_tab << "$outfile = $fileName;" << std::endl;
    *streamPtr.get() << double_tab << "} else {" << std::endl;
    *streamPtr.get() << triple_tab << "system (\"cp $fileName $file_prehip\");" << std::endl;
    *streamPtr.get() << triple_tab << "$infile = $file_prehip;" << std::endl;
    *streamPtr.get() << triple_tab << "$outfile = $fileName;" << std::endl << double_tab << "}" << std::endl;
    *streamPtr.get() << double_tab << "open(INFILE,\"<\", $infile) or die \"error: could not open $infile\";" << std::endl;
    *streamPtr.get() << double_tab << "open(OUTFILE,\">\", $outfile) or die \"error: could not open $outfile\";" << std::endl;
    *streamPtr.get() << double_tab << "$OUTFILE = OUTFILE;" << std::endl;
    *streamPtr.get() << tab << "} else {" << std::endl;
    *streamPtr.get() << double_tab << "open(INFILE,\"<\", $fileName) or die \"error: could not open $fileName\";" << std::endl;
    *streamPtr.get() << double_tab << "$OUTFILE = STDOUT;" << std::endl << tab << "}" << std::endl;
    *streamPtr.get() << tab << "# Count of transforms in this file" << std::endl;
    *streamPtr.get() << tab << "clearStats(\\%ft, \\@statNames);" << std::endl;
    *streamPtr.get() << tab << sMy << "$countIncludes = 0;" << std::endl;
    *streamPtr.get() << tab << sMy << "$countKeywords = 0;" << std::endl;
    *streamPtr.get() << tab << sMy << "$warnings = 0;" << std::endl;
    *streamPtr.get() << tab << sMy << "%warningTags;" << std::endl;
    *streamPtr.get() << tab << sMy << "$lineCount = 0;" << std::endl;
    *streamPtr.get() << tab << "undef $/;" << std::endl;
    *streamPtr.get() << tab << "# Read whole file at once, so we can match newlines" << std::endl;
    *streamPtr.get() << tab << "while (<INFILE>) {" << std::endl;
    *streamPtr.get() << double_tab << "$countKeywords += m/__global__/;" << std::endl;
    *streamPtr.get() << double_tab << "$countKeywords += m/__shared__/;" << std::endl;
    *streamPtr.get() << double_tab << "simpleSubstitutions();" << std::endl;
    *streamPtr.get() << double_tab << "transformExternShared();" << std::endl;
    *streamPtr.get() << double_tab << "transformKernelLaunch();" << std::endl;
    *streamPtr.get() << double_tab << "if ($print_stats) {" << std::endl;
    *streamPtr.get() << triple_tab << "while (/(\\b(hip|HIP)([A-Z]|_)\\w+\\b)/g) {" << std::endl;
    *streamPtr.get() << quad_tab << "$convertedTags{$1}++;" << std::endl;
    *streamPtr.get() << triple_tab << "}" << std::endl;
    *streamPtr.get() << double_tab << "}" << std::endl;
    *streamPtr.get() << double_tab << sMy << "$hasDeviceCode = $countKeywords + $ft{'device_function'};" << std::endl;
    *streamPtr.get() << double_tab << "unless ($quiet_warnings) {" << std::endl;
    *streamPtr.get() << triple_tab << "# Copy into array of lines, process line-by-line to show warnings" << std::endl;
    *streamPtr.get() << triple_tab << "if ($hasDeviceCode or (/\\bcu|CU/) or (/<<<.*>>>/)) {" << std::endl;
    *streamPtr.get() << quad_tab << sMy << "@lines = split /\\n/, $_;" << std::endl;
    *streamPtr.get() << quad_tab << "# Copy the whole file" << std::endl;
    *streamPtr.get() << quad_tab << sMy << "$tmp = $_;" << std::endl;
    *streamPtr.get() << quad_tab << sMy << "$line_num = 0;" << std::endl;
    *streamPtr.get() << quad_tab << "foreach (@lines) {" << std::endl;
    *streamPtr.get() << tab_5 << "$line_num++;" << std::endl;
    *streamPtr.get() << tab_5 << "# Remove any whitelisted words" << std::endl;
    *streamPtr.get() << tab_5 << "foreach $w (@whitelist) {" << std::endl;
    *streamPtr.get() << tab_6 << "s/\\b$w\\b/ZAP/" << std::endl;
    *streamPtr.get() << tab_5 << "}" << std::endl;
    *streamPtr.get() << tab_5 << sMy << "$tag;" << std::endl;
    *streamPtr.get() << tab_5 << "if ((/(\\bcuda[A-Z]\\w+)/) or (/<<<.*>>>/)) {" << std::endl;
    *streamPtr.get() << tab_6 << "# Flag any remaining code that look like cuda API calls: may want to add these to hipify" << std::endl;
    *streamPtr.get() << tab_6 << "$tag = (defined $1) ? $1 : \"Launch\";" << std::endl;
    *streamPtr.get() << tab_5 << "}" << std::endl;
    *streamPtr.get() << tab_5 << "if (defined $tag) {" << std::endl;
    *streamPtr.get() << tab_6 << "$warnings++;" << std::endl;
    *streamPtr.get() << tab_6 << "$warningTags{$tag}++;" << std::endl;
    *streamPtr.get() << tab_6 << "print STDERR \"  warning: $fileName:#$line_num : $_\\n\";" << std::endl;
    *streamPtr.get() << tab_5 << "}" << std::endl;
    *streamPtr.get() << tab_5 << "$s = warnUnsupportedDeviceFunctions($line_num);" << std::endl;
    *streamPtr.get() << tab_5 << "$warnings += $s;" << std::endl;
    *streamPtr.get() << quad_tab << "}" << std::endl;
    *streamPtr.get() << quad_tab << "$_ = $tmp;" << std::endl;
    *streamPtr.get() << triple_tab << "}" << std::endl;
    *streamPtr.get() << double_tab << "}" << std::endl;
    *streamPtr.get() << double_tab << "if ($hasDeviceCode > 0) {" << std::endl;
    *streamPtr.get() << triple_tab << "$ft{'device_function'} += countSupportedDeviceFunctions();" << std::endl;
    *streamPtr.get() << double_tab << "}" << std::endl;
    *streamPtr.get() << double_tab << "transformHostFunctions();" << std::endl;
    *streamPtr.get() << double_tab << "# TODO: would like to move this code outside loop but it uses $_ which contains the whole file" << std::endl;
    *streamPtr.get() << double_tab << "unless ($no_output) {" << std::endl;
    *streamPtr.get() << triple_tab << sConv << std::endl;
    *streamPtr.get() << triple_tab << sMy << "$kernStuff  = $hasDeviceCode + $ft{'" << counterNames[CONV_KERNEL_LAUNCH] << "'} + $ft{'" << counterNames[CONV_DEVICE_FUNC] << "'};" << std::endl;
    *streamPtr.get() << triple_tab << sMy << "$totalCalls = $apiCalls + $kernStuff;" << std::endl;
    *streamPtr.get() << triple_tab << "$is_dos = m/\\r\\n$/;" << std::endl;
    *streamPtr.get() << triple_tab << "if ($totalCalls and ($countIncludes == 0) and ($kernStuff != 0)) {" << std::endl;
    *streamPtr.get() << quad_tab << "# TODO: implement hipify-clang's logic with header files AMAP" << std::endl;
    *streamPtr.get() << quad_tab << "print $OUTFILE '#include \"hip/hip_runtime.h\"' . ($is_dos ? \"\\r\\n\" : \"\\n\");" << std::endl;
    *streamPtr.get() << triple_tab << "}" << std::endl;
    *streamPtr.get() << triple_tab << "print $OUTFILE  \"$_\";" << std::endl;
    *streamPtr.get() << double_tab << "}" << std::endl;
    *streamPtr.get() << double_tab << "$lineCount = $_ =~ tr/\\n//;" << std::endl;
    *streamPtr.get() << tab << "}" << std::endl;
    *streamPtr.get() << tab << sMy << "$totalConverted = totalStats(\\%ft);" << std::endl;
    *streamPtr.get() << tab << "if (($totalConverted+$warnings) and $print_stats) {" << std::endl;
    *streamPtr.get() << double_tab << "printStats(\"  info: converted\", \\@statNames, \\%ft, $warnings, $lineCount);" << std::endl;
    *streamPtr.get() << double_tab << "print STDERR \" in '$fileName'\\n\";" << std::endl;
    *streamPtr.get() << tab << "}" << std::endl;
    *streamPtr.get() << tab << "# Update totals for all files" << std::endl;
    *streamPtr.get() << tab << "addStats(\\%tt, \\%ft);" << std::endl;
    *streamPtr.get() << tab << "$Twarnings += $warnings;" << std::endl;
    *streamPtr.get() << tab << "$TlineCount += $lineCount;" << std::endl;
    *streamPtr.get() << tab << "foreach $key (keys %warningTags) {" << std::endl;
    *streamPtr.get() << double_tab << "$TwarningTags{$key} += $warningTags{$key};" << std::endl;
    *streamPtr.get() << tab << "}" << std::endl;
    *streamPtr.get() << "}" << std::endl;
    *streamPtr.get() << "# Print total stats for all files processed:" << std::endl;
    *streamPtr.get() << "if ($print_stats and ($fileCount > 1)) {" << std::endl;
    *streamPtr.get() << tab << "print STDERR \"\\n\";" << std::endl;
    *streamPtr.get() << tab << "printStats(\"  info: TOTAL-converted\", \\@statNames, \\%tt, $Twarnings, $TlineCount);" << std::endl;
    *streamPtr.get() << tab << "print STDERR \"\\n\";" << std::endl;
    *streamPtr.get() << tab << "foreach my $key (sort { $TwarningTags{$b} <=> $TwarningTags{$a} } keys %TwarningTags) {" << std::endl;
    *streamPtr.get() << double_tab << "printf STDERR \"  warning: unconverted %s : %d\\n\", $key, $TwarningTags{$key};" << std::endl;
    *streamPtr.get() << tab << "}" << std::endl;
    *streamPtr.get() << tab << sMy << "$kernelCnt = keys %Tkernels;" << std::endl;
    *streamPtr.get() << tab << "printf STDERR \"  kernels (%d total) : \", $kernelCnt;" << std::endl;
    *streamPtr.get() << tab << "foreach my $key (sort { $Tkernels{$b} <=> $Tkernels{$a} } keys %Tkernels) {" << std::endl;
    *streamPtr.get() << double_tab << "printf STDERR \"  %s(%d)\", $key, $Tkernels{$key};" << std::endl;
    *streamPtr.get() << tab << "}" << std::endl;
    *streamPtr.get() << tab << "print STDERR \"\\n\\n\";" << std::endl;
    *streamPtr.get() << "}" << std::endl;
    *streamPtr.get() << "if ($print_stats) {" << std::endl;
    *streamPtr.get() << tab << "foreach my $key (sort { $convertedTags{$b} <=> $convertedTags{$a} } keys %convertedTags) {" << std::endl;
    *streamPtr.get() << double_tab << "printf STDERR \"  %s %d\\n\", $key, $convertedTags{$key};" << std::endl;
    *streamPtr.get() << tab << "}" << std::endl << "}" << std::endl;
    streamPtr.get()->flush();
    bool ret = true;
    EC = sys::fs::copy_file(tmpFile, dstHipifyPerl);
    if (EC) {
      llvm::errs() << "\n" << sHipify << sError << EC.message() << ": while copying " << tmpFile << " to " << dstHipifyPerl << "\n";
      ret = false;
    }
    if (!SaveTemps) sys::fs::remove(tmpFile);
    return ret;
  }
}
