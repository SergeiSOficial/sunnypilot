#include <cmath>
extern "C" void run(const float*x,const float*w,const float*b,float*y,int k,int n){for(int o=0;o<n;o++){float sum=0;for(int i=0;i<k;i++)sum=std::fma(x[i],w[o*k+i],sum);y[o]=sum+b[o];}}
