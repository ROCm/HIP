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
  const std::string sSub = "sub";
  const std::string sReturn_0 = "return 0;\n";
  const std::string sReturn_m = "return $m;\n";
  const std::string sForeach = "foreach $func (\n";
  const std::string sMy = "my $m = 0;\n";
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
    *streamPtr.get() << "my $whitelist = \"\";" << std::endl << std::endl;
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
    *streamPtr.get() << tab << "my %count = %{ shift() };" << std::endl;
    *streamPtr.get() << tab << "my $total = 0;" << std::endl;
    *streamPtr.get() << tab << "foreach $key (keys %count) {" << std::endl;
    *streamPtr.get() << double_tab << "$total += $count{$key};" << std::endl << tab << "}" << std::endl;
    *streamPtr.get() << tab << "return $total;" << std::endl << "};" << std::endl;
    *streamPtr.get() << std::endl << sSub << " printStats" << " {" << std::endl;
    *streamPtr.get() << tab << "my $label     = shift();" << std::endl;
    *streamPtr.get() << tab << "my @statNames = @{ shift() };" << std::endl;
    *streamPtr.get() << tab << "my %counts    = %{ shift() };" << std::endl;
    *streamPtr.get() << tab << "my $warnings  = shift();" << std::endl;
    *streamPtr.get() << tab << "my $loc       = shift();" << std::endl;
    *streamPtr.get() << tab << "my $total     = totalStats(\\%counts);" << std::endl;
    *streamPtr.get() << tab << "printf STDERR \"%s %d CUDA->HIP refs ( \", $label, $total;" << std::endl;
    *streamPtr.get() << tab << "foreach $stat (@statNames) {" << std::endl;
    *streamPtr.get() << double_tab << "printf STDERR \"%s:%d \", $stat, $counts{$stat};" << std::endl;
    *streamPtr.get() << tab << "}" << std::endl;
    *streamPtr.get() << tab << "printf STDERR \")\\n  warn:%d LOC:%d\", $warnings, $loc;" << std::endl << "};" << std::endl;
    for (int i = 0; i < 2; ++i) {
      *streamPtr.get() << std::endl << sSub << " " << (i ? "clearStats" : "addStats") << " {" << std::endl;
      *streamPtr.get() << tab << "my $dest_ref  = shift();" << std::endl;
      *streamPtr.get() << tab << (i ? "my @statNames = @{ shift() };" : "my %adder     = %{ shift() };") << std::endl;
      *streamPtr.get() << tab << "foreach " << (i ? "$stat(@statNames)" : "$key (keys %adder)") << " {" << std::endl;
      *streamPtr.get() << double_tab << "$dest_ref->" << (i ? "{$stat} = 0;" : "{$key} += $adder{$key};") << std::endl << tab << "}" << std::endl << "};" << std::endl;
    }
  }

  void generateHostFunctions(std::unique_ptr<std::ostream>& streamPtr) {
    *streamPtr.get() << std::endl << sSub << " transformHostFunctions" << "{" << std::endl << tab << sMy;
    std::set<std::string> &funcSet = DeviceSymbolFunctions0;
    const std::string s0 = "$m += s/(?<!\\/\\/ CHECK: )($func)\\s*\\(\\s*([^,]+)\\s*,/$func\\(";
    const std::string s1 = "$m += s/(?<!\\/\\/ CHECK: )($func)\\s*\\(\\s*([^,]+)\\s*,\\s*([^,\\)]+)\\s*(,\\s*|\\))\\s*/$func\\($2, ";
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
    *streamPtr.get() << tab << sReturn_m << "}" << std::endl;
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
    std::string sCommon = tab + sMy + tab + sForeach;
    subCountSupported << std::endl << sSub << " countSupportedDeviceFunctions" << " {" << std::endl << (countSupported ? sCommon : tab + sReturn_0);
    subWarnUnsupported << std::endl << sSub << " warnUnsupportedDeviceFunctions" << " {" << std::endl << (countUnsupported ? tab + "my $line_num = shift;\n" + sCommon : tab + sReturn_0);
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
      subCommon << double_tab << "my $mt_namespace = m/(\\w+)::($func)\\s*\\(\\s*.*\\s*\\)/g;" << std::endl;
      subCommon << double_tab << "my $mt = m/($func)\\s*\\(\\s*.*\\s*\\)/g;" << std::endl;
      subCommon << double_tab << "if ($mt && !$mt_namespace) {" << std::endl;
      subCommon << triple_tab << "$m += $mt;" << std::endl;
    }
    if (countSupported) {
      subCountSupported << subCommon.str();
    }
    if (countUnsupported) {
      subWarnUnsupported << subCommon.str();
      subWarnUnsupported << triple_tab << "print STDERR \"  warning: $fileName:$line_num: unsupported device function \\\"$func\\\": $_\\n\";" << std::endl;
    }
    if (countSupported || countUnsupported) {
      sCommon = double_tab + "}\n" + tab + "}\n" + tab + sReturn_m;
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
    std::string dstPerlMap = OutputPerlMapFilename, dstPerlMapDir = OutputPerlMapDir;
    if (dstPerlMap.empty()) dstPerlMap = "hipify-perl-map";
    std::error_code EC;
    if (!dstPerlMapDir.empty()) {
      std::string sOutputPerlMapDirAbsPath = getAbsoluteDirectoryPath(OutputPerlMapDir, EC, "output hipify-perl map");
      if (EC) return false;
      dstPerlMap = sOutputPerlMapDirAbsPath + "/" + dstPerlMap;
    }
    SmallString<128> tmpFile;
    StringRef ext = "hipify-tmp";
    EC = sys::fs::createTemporaryFile(dstPerlMap, ext, tmpFile);
    if (EC) {
      llvm::errs() << "\n" << sHipify << sError << EC.message() << ": " << tmpFile << "\n";
      return false;
    }
    std::unique_ptr<std::ostream> streamPtr = std::unique_ptr<std::ostream>(new std::ofstream(tmpFile.c_str(), std::ios_base::trunc));
    generateHeader(streamPtr);
    std::string sConv = "my $conversions = ";
    *streamPtr.get() << "@statNames = (";
    for (int i = 0; i < NUM_CONV_TYPES - 1; ++i) {
      *streamPtr.get() << "\"" << counterNames[i] << "\", ";
      sConv += "$ft{'" + std::string(counterNames[i]) + "'} + ";
    }
    *streamPtr.get() << "\"" << counterNames[NUM_CONV_TYPES - 1] << "\");" << std::endl;
    generateStatFunctions(streamPtr);
    *streamPtr.get() << std::endl << sConv << "$ft{'" << counterNames[NUM_CONV_TYPES - 1] << "'};" << std::endl << std::endl;
    for (int i = 0; i < NUM_CONV_TYPES; ++i) {
      if (i == CONV_INCLUDE_CUDA_MAIN_H || i == CONV_INCLUDE) {
        for (auto& ma : CUDA_INCLUDE_MAP) {
          if (Statistics::isUnsupported(ma.second)) continue;
          if (i == ma.second.type) {
            std::string sCUDA = ma.first.str();
            std::string sHIP = ma.second.hipName.str();
            sCUDA = std::regex_replace(sCUDA, std::regex("/"), "\\/");
            sHIP = std::regex_replace(sHIP, std::regex("/"), "\\/");
            *streamPtr.get() << "$ft{'" << counterNames[ma.second.type] << "'} += s/\\b" << sCUDA << "\\b/" << sHIP << "/g;" << std::endl;
          }
        }
      }
      else {
        for (auto& ma : CUDA_RENAMES_MAP()) {
          if (Statistics::isUnsupported(ma.second)) continue;
          if (i == ma.second.type) {
            *streamPtr.get() << "$ft{'" << counterNames[ma.second.type] << "'} += s/\\b" << ma.first.str() << "\\b/" << ma.second.hipName.str() << "/g;" << std::endl;
          }
        }
      }
    }
    generateHostFunctions(streamPtr);
    generateDeviceFunctions(streamPtr);
    streamPtr.get()->flush();
    bool ret = true;
    EC = sys::fs::copy_file(tmpFile, dstPerlMap);
    if (EC) {
      llvm::errs() << "\n" << sHipify << sError << EC.message() << ": while copying " << tmpFile << " to " << dstPerlMap << "\n";
      ret = false;
    }
    if (!SaveTemps) sys::fs::remove(tmpFile);
    return ret;
  }
}
