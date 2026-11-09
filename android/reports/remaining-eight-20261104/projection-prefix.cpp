#include <cmath>
extern "C" void run(const float*x,const float*w,const float*b,float*y,int k,int n,int s){for(int o=0;o<n;o++)for(int p=0;p<s;p++){float sum=0;for(int i=0;i<k;i++)sum=std::fma(x[i*s+p],w[o*k+i],sum);y[o*s+p]=b?sum+b[o]:sum;}}
