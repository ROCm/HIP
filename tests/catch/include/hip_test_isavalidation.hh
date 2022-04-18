#include <fstream>
#include <filesystem>
#include <regex>
#include <algorithm>

namespace HipTest {

static bool assemblyFile_Verification(
    std::string filename,
    std::vector<std::string> insts,
    std::string funcname,
    std::string path="./catch/unit/deviceLib/") {

  std::stringstream sstart;
  sstart << "Begin function .*";
  sstart << funcname;
  std::regex start(sstart.str());

  std::stringstream sstop;
  sstop << ".Lfunc_end\\d+-.*";
  sstop << funcname;
  std::regex stop = std::regex(sstop.str());

  std::filesystem::path dir = std::filesystem::path(path);
  REQUIRE(std::filesystem::is_directory(dir));
  // glob for 90a
  bool found = false;
  std::filesystem::path isa;
  for (const auto& path : std::filesystem::directory_iterator(dir)) {
    if (std::regex_search(path.path().string(), std::regex(filename))) {
      // found
      found = true;
      isa = path;
      break;
    }
  }
  REQUIRE(found);

  std::ifstream file(isa);
  REQUIRE(file.is_open());

  std::vector<std::regex> rinsts;
  for (const auto& inst : insts) {
    rinsts.push_back(std::regex(inst));
  }

  std::vector<bool> results(insts.size(), false);
  std::string line;
  bool started = false;
  bool stopped = false;
  while (getline(file, line)) {
    if (std::regex_search(line, start)) {
      started = true;
    } else if (std::regex_search(line, stop)) {
      stopped = true;
    } else if (started) {
      for (size_t i = 0; i < rinsts.size(); ++i) {
        if (std::regex_search(line, rinsts[i])) {
          results[i] = true;
          break;
        }
      }
    }
  }
  REQUIRE(stopped);
  return std::all_of(results.cbegin(), results.cend(), [](bool x){ return x; });

}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"

static bool assemblyFile_Verification(
    std::string filename,
    std::string insts,
    std::string funcname,
    std::string path="./catch/unit/deviceLib/") {
  std::vector<std::string> vinsts = {insts};
  return assemblyFile_Verification(
    filename,
    vinsts,
    funcname,
    path);
}

#pragma GCC diagnostic pop

}
