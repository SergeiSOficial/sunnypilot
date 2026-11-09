#include <math.h>
float cube(float x){float s=x*x,e=fmaf(x,x,-s),c=s*x,t=fmaf(s,x,-c),r=fmaf(e,x,t);return c+r;}
