#include <hc.hpp>

int main(int argc, char *argv[])
{
    int size = 1000000;
    bool pass = true;

    // Allocate auto-managed host/device views of data:
    hc::array_view<float> A(size);
    hc::array_view<float> B(size);
    hc::array_view<float> C(size);

    // Initialize host data
    for (int i=0; i<size; i++) {
        A[i] = 1.618f * i; 
        B[i] = 3.142f * i;
    }
    C.discard_data(); // tell runtime not to copy CPU host data.


    // Launch kernel onto default accelerator:
    hc::parallel_for_each(hc::extent<1> (size),
      [=] (hc::index<1> idx) [[hc]] { 
        int i = idx[0];
        C[i] = A[i] + B[i];
    });

    for (int i=0; i<size; i++) {
        float ref= 1.618f * i + 3.142f * i;
        if (C[i] != ref) {
            printf ("error:%d computed=%6.2f, reference=%6.2f\n", i, C[i], ref);
            pass = false;
        }
    };
    if (pass) printf ("PASSED!\n");
}
