#include <cmath>
#include <cstring>
static float high(float x){unsigned u;std::memcpy(&u,&x,4);u&=0xfffff000u;std::memcpy(&x,&u,4);return x;}
static float compensated(float a,float b,float c){float ah=high(a),bh=high(b),al=a-ah,bl=b-bh;float p=a*b;float e=((ah*bh-p)+ah*bl+al*bh)+al*bl;float s=p+c;float v=s-p;float se=(p-(s-v))+(c-v);return s+(se+e);}
extern "C" void run(const float*x,const float*w,const float*b,float*y,int k,int n,int s,int mode){for(int o=0;o<n;o++)for(int p=0;p<s;p++){float sum=0;for(int i=0;i<k;i++){if(mode==0)sum=std::fma(x[i*s+p],w[o*k+i],sum);else if(mode==2)sum=compensated(x[i*s+p],w[o*k+i],sum);else {volatile float product=x[i*s+p]*w[o*k+i];sum=sum+product;}}y[o*s+p]=b?sum+b[o]:sum;}}
