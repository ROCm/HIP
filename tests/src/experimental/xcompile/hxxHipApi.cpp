#include"gxxHipApi.h"
#include<vector>
#include"hip_runtime.h"

#define LEN 1024*1024
#define SIZE LEN * sizeof(float)

class memManager;

int main()
{
    std::vector<class memManager> Vec(4);
    for(int i=0;i<Vec.size();i++){
        Vec[i] = memManager(SIZE);
    }

    for(int i=0;i<4;i++)
    {
        Vec[i].setHstPtr(new float[LEN]);
    }

    for(int i=0;i<2;i++)
    {
        Vec[i].memAlloc<float>();
    }

    for(int i=0;i<2;i++)
    {
        Vec[i].hostMemSet((i+1)*1.0f);
        Vec[i].H2D();
    }

    Vec[2].setDevPtr(Vec[0].getDevPtr<float>());
    Vec[3].setDevPtr(Vec[1].getDevPtr<float>());

    for(int i=2;i<Vec.size();i++)
    {
        Vec[i].D2H();
    }
    
    assert(Vec[0].getHstPtr<float>()[10] == Vec[2].getHstPtr<float>()[10]);
    assert(Vec[1].getHstPtr<float>()[10] == Vec[3].getHstPtr<float>()[10]);
}
