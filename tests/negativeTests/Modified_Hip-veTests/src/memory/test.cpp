#include<iostream>
#include<string>
#include<hip/hip_runtime.h>
#include<hip/hip_runtime_api.h>

using namespace std;

int main()
{
string out;
void *dst;
int val[10];

out=hipGetErrorString(hipMalloc(&dst,80));
cout<<"The value of hipMalloc(&dst,80) is: "<<out<<endl;

//out=hipGetErrorString(hipMemset(dst,99,80));

//out=hipGetErrorString(hipMemset(NULL,99,80));

//out=hipGetErrorString(hipMemset(0,99,80));
//cout<<"The output of hipMemset() is: "<<out<<endl;

//out=hipGetErrorString(hipMemset(dst,NULL,80));

//providing out of bound memory value

out=hipGetErrorString(hipMemset(dst,99,81));

cout<<"The output of hipMemset(dst,99,81) is: "<<out<<endl;


/*out=hipGetErrorString(hipMemcpy(val,dst,sizeof(val),hipMemcpyDeviceToHost));
cout<<"The output of hipMemcpy() is: "<<out<<endl;

cout<<"The content from gpu is: "<<val[0]<<endl;
*/

}
