#include "timer.h"

#include <stdlib.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define VC_EXTRALEAN
#include <windows.h>
#pragma comment(lib, "user32")
#endif

#ifdef __linux__
#include <time.h>
#define NANOSECONDS_PER_SEC 1000000000
#endif

CPerfCounter::CPerfCounter() : _clocks(0), _start(0)
{

#ifdef _WIN32

    QueryPerformanceFrequency((LARGE_INTEGER *)&_freq);

#endif

#ifdef __linux__
    _freq = NANOSECONDS_PER_SEC;
#endif

}

CPerfCounter::~CPerfCounter()
{
    // EMPTY!
}

void
CPerfCounter::Start(void)
{

#ifdef _WIN32

    if( _start )
    {
        MessageBox(NULL, "Bad Perf Counter Start", "Error", MB_OK);
        exit(0);
    }
    QueryPerformanceCounter((LARGE_INTEGER *)&_start);

#endif
#ifdef __linux__

    struct timespec s;
    clock_gettime(CLOCK_MONOTONIC, &s);
    _start = (i64)s.tv_sec * NANOSECONDS_PER_SEC + (i64)s.tv_nsec ;

#endif

}

void
CPerfCounter::Stop(void)
{
    i64 n;

#ifdef _WIN32

    if( !_start )
    {
        MessageBox(NULL, "Bad Perf Counter Stop", "Error", MB_OK);
        exit(0);
    }

    QueryPerformanceCounter((LARGE_INTEGER *)&n);

#endif
#ifdef __linux__

    struct timespec s;
    clock_gettime(CLOCK_MONOTONIC, &s);
    n = (i64)s.tv_sec * NANOSECONDS_PER_SEC + (i64)s.tv_nsec ;

#endif

    n -= _start;
    _start = 0;
    _clocks += n;
}

void
CPerfCounter::Reset(void)
{

#ifdef _WIN32
    if( _start )
    {
        MessageBox(NULL, "Bad Perf Counter Reset", "Error", MB_OK);
        exit(0);
    }
#endif
    _clocks = 0;
}

double
CPerfCounter::GetElapsedTime(void)
{
#ifdef _WIN32
    if( _start ) {
        MessageBox(NULL, "Trying to get time while still running.", "Error", MB_OK);
        exit(0);
    }
#endif

    return (double)_clocks / (double)_freq;

}
