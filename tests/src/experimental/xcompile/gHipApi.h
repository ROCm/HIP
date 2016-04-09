#ifndef GHIPAPI_H
#define GHIPAPI_H

#include<stdlib.h>

typedef struct {
void *hst_ptr;
void *dev_ptr;
size_t size;
void (*h2d)();
void (*d2h)();
void (*malloc_hip)();
void (*malloc_hst)();
} mem_manager;

mem_manager *mem_manager_start(size_t);

void memset_hst(mem_manager*, float);

#endif
