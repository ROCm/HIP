#include <iostream>
#include <iomanip>
#include <string>

#define CASE_STR(x) case x: return #x;


// Building block functions:
template <typename T>
std::string ToHexString(T v)
{
    std::ostringstream ss;
    ss << "0x" << std::hex << v;
    return ss.str();
};



// Template overloads for ToString to handle various types:
// Note these use C++11 variadic templates
template <typename T>
std::string ToString(T v) {
    std::ostringstream ss;
    ss << v;
    return ss.str();
};


#if 0
template <>
std::string ToString(void* v) {
    std::ostringstream ss;
    //ss << "0x" << std::setw(16) << std::setfill('0') << std::hex << v;
    ss << "0x" << std::hex << v;
    return ss.str();
};
#endif


template <>
std::string ToString(hipMemcpyKind v) {
    switch(v) {
    CASE_STR(hipMemcpyHostToHost);
    CASE_STR(hipMemcpyHostToDevice);
    CASE_STR(hipMemcpyDeviceToHost);
    CASE_STR(hipMemcpyDeviceToDevice);
    CASE_STR(hipMemcpyDefault);
    default : return ToHexString(v);
    };
};


template <>
std::string ToString(hipError_t v) {
    return ihipErrorString(v);
};


// Catch empty arguments case
std::string ToString() {
    return ("");
}

    ///
// C++11 variadic template - peels off first argument, converts to string, and calls itself again to peel the next arg.
template <typename T, typename... Args> 
std::string ToString(T first, Args... args) {
    return ToString(first) + ", " + ToString(args...) ;
}


