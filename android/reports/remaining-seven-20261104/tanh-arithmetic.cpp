#include <cmath>
#include <cstring>
static float high(float x){unsigned u;std::memcpy(&u,&x,4);u&=0xfffff000u;std::memcpy(&x,&u,4);return x;}
static float mad(float a,float b,float c,int m){if(m==0)return std::fma(a,b,c); if(m==1){volatile float p=a*b;return p+c;}float ah=high(a),bh=high(b),al=a-ah,bl=b-bh,p=a*b,e=((ah*bh-p)+ah*bl+al*bh)+al*bl,s=p+c,v=s-p,se=(p-(s-v))+(c-v);return s+(se+e);}
extern "C" void run(const float*x,float*y,int n,int m){for(int i=0;i<n;i++){float v=std::fmin(9.f,std::fmax(-9.f,x[i])),q=v*v,a=-2.76076847742355e-16f;
a=mad(a,q,2.00018790482477e-13f,m);a=mad(a,q,-8.60467152213735e-11f,m);a=mad(a,q,5.12229709037114e-08f,m);a=mad(a,q,1.48572235717979e-05f,m);a=mad(a,q,6.37261928875436e-04f,m);a=mad(a,q,4.89352455891786e-03f,m);
float b=1.19825839466702e-06f;b=mad(b,q,1.18534705686654e-04f,m);b=mad(b,q,2.26843463243900e-03f,m);b=mad(b,q,4.89352518554385e-03f,m);y[i]=(a*v)/b;}}
