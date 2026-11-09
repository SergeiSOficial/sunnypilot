#include <cmath>
#include <algorithm>
extern "C" void run(const float* input,float* output,int n) {
 for(int i=0;i<n;i++) {
 float x=std::clamp(input[i],-9.f,9.f); float s=x*x;
 float a=-2.76076847742355e-16f;
 a=std::fma(a,s,2.00018790482477e-13f); a=std::fma(a,s,-8.60467152213735e-11f);
 a=std::fma(a,s,5.12229709037114e-08f); a=std::fma(a,s,1.48572235717979e-05f);
 a=std::fma(a,s,6.37261928875436e-04f); a=std::fma(a,s,4.89352455891786e-03f);
 float b=1.19825839466702e-06f; b=std::fma(b,s,1.18534705686654e-04f);
 b=std::fma(b,s,2.26843463243900e-03f); b=std::fma(b,s,4.89352518554385e-03f);
 float product=a*x; output[i]=product/b;
 }
}
