#ifndef GXXHIPAPI_H
#define GXXHIPAPI_H

#include<stdlib.h>
#include"hip_runtime_api.h"

class memManager{
private:
    void* devPtr;
    void* hstPtr;
    size_t size;
public:
    memManager(size_t size) : size(size) {}
    memManager(){}
    memManager(const memManager &obj);
    template<typename T>
    void setDevPtr(T* ptr)
    {
        devPtr = (void*)ptr;
    }

    template<typename T>
    T* getDevPtr()
    {
         return (T*)devPtr;
    }

    template<typename T>
    void setHstPtr(T* ptr)
    {
         hstPtr = (void*)ptr;
    }

    template<typename T>
    T* getHstPtr()
    {
         return (T*)hstPtr;
    }

    void H2D();
    void D2H();
    template<typename T>
    void hostMemSet(T val)
    {
        T* tmpPtr = (T*)hstPtr;
        for(int i=0;i<size/sizeof(T);i++)
        {
            tmpPtr[i] = val;
        }
    }
    template<typename T>
    void memAlloc()
    {
          hipMalloc((void**)&devPtr, size);
    }
};

#endif
