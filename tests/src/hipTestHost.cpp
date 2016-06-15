#include "test_common.h"
#include <iostream>
#include "hip_runtime.h"
#include "hip_runtime_api.h"

#define N 512

bool check_erfcinvf(){
    uint32_t len = 4;
    float Val[] = {0.1, 1.2, 1, 0.9};
    float Out[] = {1.16309, -0.179144, 0, 0.0889};
    for(int i=0;i<len;i++){
        if(Out[i] - erfcinvf(Val[i]) > 0.0001)
        {
            return false;
        }
    }
    return true;
}

bool check_erfcxf(){
    uint32_t len = 4;
    float Val[] = {-0.5, 15, 3.2, 1};
    float Out[] = {1.9524, 0.0375, 0.1687, 0.4276};
    for(int i=0;i<len;i++){
        if(Out[i] - erfcxf(Val[i]) > 0.0001)
        {
            return false;
        }
    }
    return true;
}

bool check_erfinvf()
{
    uint32_t len = 4;
    float Val[] = {0, -0.5, 0.9, -0.2};
    float Out[] = {0, -0.4769, 1.1631, -0.1791};
    for(int i=0;i<len;i++){
        if(Out[i] - erfinvf(Val[i]) > 0.0001){
            return false;
        }
    }
    return true;
}

int main(){
    float *Af = new float[N];
    double *A = new double[N];
    for(int i=0;i<N;i++){
        Af[i] = i * 1.0f;
        A[i] = i * 1.0;
    }
    if(check_erfcinvf() && check_erfcxf() && check_erfcinvf()){
        passed();
    }

}
