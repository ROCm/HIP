#include"gHipApi.h"
#include"stdio.h"
#include "assert.h"
#define LEN 1024*1024
#define SIZE LEN * sizeof(float)

int main()
{
    mem_manager *a;
    a = mem_manager_start(SIZE);
    a->malloc_hst(a);
    a->malloc_hip(a);
    memset_hst(a, 1.0f);
    a->h2d(a);
    memset_hst(a, 0.0f);
    a->d2h(a);
    assert(((float*)a->hst_ptr)[10] == 1.0f);
}



