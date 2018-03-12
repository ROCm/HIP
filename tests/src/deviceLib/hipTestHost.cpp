/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.
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
#include "test_common.h"
#include <iostream>
#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"

#define N 512

bool check_erfcinvf() {
    uint32_t len = 4;
    float Val[] = {0.1, 1.2, 1, 0.9};
    float Out[] = {1.16309, -0.179144, 0, 0.0889};
    for (int i = 0; i < len; i++) {
        if (Out[i] - erfcinvf(Val[i]) > 0.0001) {
            return false;
        }
    }
    return true;
}

bool check_erfcxf() {
    uint32_t len = 4;
    float Val[] = {-0.5, 15, 3.2, 1};
    float Out[] = {1.9524, 0.0375, 0.1687, 0.4276};
    for (int i = 0; i < len; i++) {
        if (Out[i] - erfcxf(Val[i]) > 0.0001) {
            return false;
        }
    }
    return true;
}

bool check_erfinvf() {
    uint32_t len = 4;
    float Val[] = {0, -0.5, 0.9, -0.2};
    float Out[] = {0, -0.4769, 1.1631, -0.1791};
    for (int i = 0; i < len; i++) {
        if (Out[i] - erfinvf(Val[i]) > 0.0001) {
            return false;
        }
    }
    return true;
}

bool check_fdividef() {
    uint32_t len = 4;
    float Val[] = {0, -0.5, 0.9, -0.2};
    float Out[] = {1, -0.4769, 1.1631, -0.1791};
    for (int i = 0; i < len; i++) {
        if (Val[i] / Out[i] - fdividef(Val[i], Out[i]) > 0.0001) {
            return false;
        }
    }
    return true;
}

bool check_erfcinv() {
    uint32_t len = 4;
    double Val[] = {0.1, 1.2, 1, 0.9};
    double Out[] = {1.16309, -0.179144, 0, 0.0889};
    for (int i = 0; i < len; i++) {
        if (Out[i] - erfcinv(Val[i]) > 0.0001) {
            return false;
        }
    }
    return true;
}

bool check_erfcx() {
    uint32_t len = 4;
    double Val[] = {-0.5, 15, 3.2, 1};
    double Out[] = {1.9524, 0.0375, 0.1687, 0.4276};
    for (int i = 0; i < len; i++) {
        if (Out[i] - erfcx(Val[i]) > 0.0001) {
            return false;
        }
    }
    return true;
}

bool check_erfinv() {
    uint32_t len = 4;
    double Val[] = {0, -0.5, 0.9, -0.2};
    double Out[] = {0, -0.4769, 1.1631, -0.1791};
    for (int i = 0; i < len; i++) {
        if (Out[i] - erfinv(Val[i]) > 0.0001) {
            return false;
        }
    }
    return true;
}

bool check_fdivide() {
    uint32_t len = 4;
    double Val[] = {0, -0.5, 0.9, -0.2};
    double Out[] = {1, -0.4769, 1.1631, -0.1791};
    for (int i = 0; i < len; i++) {
        if (Val[i] / Out[i] - fdivide(Val[i], Out[i]) > 0.0001) {
            return false;
        }
    }
    return true;
}

bool check_modff() {
    uint32_t len = 4;
    float Val[] = {0, -0.5, 0.9, -0.2};
    float iPtr[] = {0, 0, 0, 0};
    float frac[] = {0, -0.5, 0.9, -0.2};
    float Out[] = {1, 1, 1, 1};
    for (int i = 0; i < len; i++) {
        if (frac[i] - modff(Val[i], Out + i) > 0.0001 && iPtr[i] == Out[i]) {
            return false;
        }
    }
    return true;
}

bool check_modf() {
    uint32_t len = 4;
    double Val[] = {0, -0.5, 0.9, -0.2};
    double iPtr[] = {0, 0, 0, 0};
    double frac[] = {0, -0.5, 0.9, -0.2};
    double Out[] = {1, 1, 1, 1};
    for (int i = 0; i < len; i++) {
        if (frac[i] - modf(Val[i], Out + i) > 0.0001 && iPtr[i] == Out[i]) {
            return false;
        }
    }
    return true;
}

bool check_nextafterf() {
    uint32_t len = 4;
    float Val[] = {0, -0.5, 0.9, -0.2};
    float iPtr[] = {0, 0, 0, 0};
    float frac[] = {0, -0.5, 0.9, -0.2};
    float Out[] = {1, 1, 1, 1};
    for (int i = 0; i < len; i++) {
        if (nextafterf(Val[i], 1) - Val[i] > 0.0001) {
            return false;
        }
    }
    return true;
}

bool check_nextafter() {
    uint32_t len = 4;
    double Val[] = {0, -0.5, 0.9, -0.2};
    double iPtr[] = {0, 0, 0, 0};
    double frac[] = {0, -0.5, 0.9, -0.2};
    double Out[] = {1, 1, 1, 1};
    for (int i = 0; i < len; i++) {
        if (nextafter(Val[i], 1) - Val[i] > 0.0001) {
            return false;
        }
    }
    return true;
}

bool check_norm3df(float* A) {
    float f = norm3df(A[0], A[1], A[2]);
    float out = sqrt(A[0] * A[0] + A[1] * A[1] + A[2] * A[2]);
    if (f - out > 0.0001) {
        return false;
    }
    return true;
}

bool check_norm3d(double* A) {
    double f = norm3d(A[0], A[1], A[2]);
    double out = sqrt(A[0] * A[0] + A[1] * A[1] + A[2] * A[2]);
    if (f - out > 0.0001) {
        return false;
    }
    return true;
}

bool check_norm4df(float* A) {
    float f = norm4df(A[0], A[1], A[2], A[3]);
    float out = sqrt(A[0] * A[0] + A[1] * A[1] + A[2] * A[2] + A[3] * A[3]);
    if (f - out > 0.0001) {
        return false;
    }
    return true;
}

bool check_norm4d(double* A) {
    double f = norm4d(A[0], A[1], A[2], A[3]);
    double out = sqrt(A[0] * A[0] + A[1] * A[1] + A[2] * A[2] + A[3] * A[3]);
    if (f - out > 0.0001) {
        return false;
    }
    return true;
}

bool check_normcdff() {
    uint32_t len = 2;
    float Val[] = {0, 1};
    float Out[] = {0.5, 0.8413};
    for (int i = 0; i < len; i++) {
        if (Out[i] - normcdff(Val[i]) > 0.0001) {
            return false;
        }
    }
    return true;
}

bool check_normcdf() {
    uint32_t len = 2;
    float Val[] = {0, 1};
    float Out[] = {0.5, 0.8413};
    for (int i = 0; i < len; i++) {
        if (Out[i] - normcdf(Val[i]) > 0.0001) {
            return false;
        }
    }
    return true;
}


bool check_normcdfinvf() {
    uint32_t len = 2;
    double Val[] = {0.5, 0.8413};
    for (int i = 0; i < len; i++) {
        if (Val[i] - normcdfinvf(normcdff(Val[i])) > 0.0001) {
            return false;
        }
    }
    return true;
}

bool check_normcdfinv() {
    uint32_t len = 2;
    double Val[] = {0.5, 0.8413};
    for (int i = 0; i < len; i++) {
        if (Val[i] - normcdfinv(normcdf(Val[i])) > 0.0001) {
            return false;
        }
    }
    return true;
}

bool check_rcbrtf() {
    float f = 1.0f;
    if (rcbrtf(f) != 1.0f) {
        return false;
    }
    return true;
}

bool check_rcbrt() {
    double f = 1.0;
    if (rcbrt(f) != 1.0) {
        return false;
    }
    return true;
}

bool check_rhypotf() {
    float f = 1.0f;
    float g = 2.0f;
    float val = rhypotf(f, g);
    float sq = f * f + g * g;
    if (1 / (val * val) - sq > 0.0001) {
        return false;
    }
    return true;
}

bool check_rhypot() {
    double f = 1.0f;
    double g = 2.0f;
    double val = rhypot(f, g);
    double sq = f * f + g * g;
    if (1 / (val * val) - sq > 0.0001) {
        return false;
    }
    return true;
}

bool check_rnorm3df(float* A) {
    float f = rnorm3df(A[0], A[1], A[2]);
    float out = sqrt(A[0] * A[0] + A[1] * A[1] + A[2] * A[2]);
    if (f - 1 / out > 0.0001) {
        return false;
    }
    return true;
}

bool check_rnorm3d(double* A) {
    double f = rnorm3d(A[0], A[1], A[2]);
    double out = sqrt(A[0] * A[0] + A[1] * A[1] + A[2] * A[2]);
    if (f - 1 / out > 0.0001) {
        return false;
    }
    return true;
}

bool check_rnorm4df(float* A) {
    float f = rnorm4df(A[0], A[1], A[2], A[3]);
    float out = sqrt(A[0] * A[0] + A[1] * A[1] + A[2] * A[2] + A[3] * A[3]);
    if (f - 1 / out > 0.0001) {
        return false;
    }
    return true;
}

bool check_rnorm4d(double* A) {
    double f = rnorm4d(A[0], A[1], A[2], A[3]);
    double out = sqrt(A[0] * A[0] + A[1] * A[1] + A[2] * A[2] + A[3] * A[3]);
    if (f - 1 / out > 0.0001) {
        return false;
    }
    return true;
}

bool check_rnormf(float* A) {
    return (rnorm3df(A[0], A[1], A[2]) - rnormf(3, A) < 0.0001) &&
           (rnorm4df(A[0], A[1], A[2], A[3]) - rnormf(4, A) < 0.0001);
}

bool check_rnorm(double* A) {
    return (rnorm3d(A[0], A[1], A[2]) - rnorm(3, A) < 0.0001) &&
           (rnorm4d(A[0], A[1], A[2], A[3]) - rnorm(4, A) < 0.0001);
}

bool check_sincospif() {
    float s1, c1, s2, c2;
    float in1 = 1, in2 = 0.5;
    sincospif(in1, &s1, &c1);
    sincospif(in2, &s2, &c2);
    if ((s1 - 0 < 0.00001) && (s2 - 1 < 0.00001) && (c1 + 1 < 0.00001) && (c2 - 0 < 0.00001)) {
        return true;
    }
    return false;
}

bool check_sincospi() {
    double s1, c1, s2, c2;
    double in1 = 1, in2 = 0.5;
    sincospi(in1, &s1, &c1);
    sincospi(in2, &s2, &c2);
    if ((s1 - 0 < 0.00001) && (s2 - 1 < 0.00001) && (c1 + 1 < 0.00001) && (c2 - 0 < 0.00001)) {
        return true;
    }
    return false;
}

int main() {
    float* Af = new float[N];
    double* A = new double[N];
    for (int i = 0; i < N; i++) {
        Af[i] = i * 1.0f;
        A[i] = i * 1.0;
    }
    if (check_erfcinvf() && check_erfcxf() && check_erfcinvf() && check_erfcinv() &&
        check_erfcx() && check_erfcinv() && check_fdividef() && check_fdivide() && check_modff() &&
        check_modf() && check_nextafterf() && check_norm3df(Af) && check_norm3d(A) &&
        check_norm4df(Af) && check_norm4d(A) && check_normcdff() && check_normcdf() &&
        check_normcdfinvf() && check_normcdfinv() && check_rcbrtf() && check_rcbrt() &&
        check_rhypotf() && check_rhypot() && check_rnorm3df(Af) && check_rnorm3d(A) &&
        check_rnorm4df(Af) && check_rnorm4d(A) && check_rnormf(Af) && check_rnorm(A) &&
        check_sincospif() && check_sincospi()) {
        passed();
    }
}
