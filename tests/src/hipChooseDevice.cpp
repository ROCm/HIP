#include <stdio.h>
#include <hip_runtime.h>
int main( void ) {
    hipDeviceProp_t  prop;
    int dev;

    hipGetDevice( &dev ) ;
    printf( "ID of current HIP device:  %d\n", dev );

    memset( &prop, 0, sizeof( hipDeviceProp_t ) );
    prop.major = 1;
    prop.minor = 3;
    hipChooseDevice( &dev, &prop );
    printf( "ID of hip device closest to revision 1.3:  %d\n", dev );

    hipSetDevice( dev );
}
