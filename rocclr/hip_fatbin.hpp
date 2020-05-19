#ifndef HIP_FAT_BINARY_HPP
#define HIP_FAT_BINARY_HPP

namespace hip {

class FatBinaryMetaInfo {
public:
  FatBinaryMetaInfo(bool built, const void* binary_ptr, size_t binary_size):
                    built_(built), binary_ptr_(binary_ptr), binary_size_(binary_size) {}
  ~FatBinaryMetaInfo() {}

  //Set once the mod has been built
  void set_built() { built_ = true; }

  //Accessor for private vars
  bool built() const { return built_; }
  const void* binary_ptr() const { return binary_ptr_; }
  size_t binary_size() const { return binary_size_; }
private:
  bool built_;              //Set when mod is built. Used in Lazy Binary
  const void* binary_ptr_;  //Binary image ptr
  size_t binary_size_;      //Binary Size
};

typedef std::vector<std::pair<hipModule_t, FatBinaryMetaInfo*>> FatBinaryInfoType;

}; /* namespace hip */

#endif /* HIP_FAT_BINARY_HPP */
