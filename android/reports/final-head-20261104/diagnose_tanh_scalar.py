"""Reproduce first captured tanh discrepancy without changing golden tolerances."""
import ctypes,json
from pathlib import Path
import numpy as np
f=np.float32
lib=ctypes.CDLL('/usr/lib/libSystem.B.dylib');lib.fmaf.argtypes=[ctypes.c_float]*3;lib.fmaf.restype=ctypes.c_float

def fma(a,b,c):return f(lib.fmaf(float(a),float(b),float(c)))
def split(a,b,c):
 a,b,c=map(f,(a,b,c));ah=(a.view(np.uint32)&np.uint32(0xfffff000)).view(np.float32);bh=(b.view(np.uint32)&np.uint32(0xfffff000)).view(np.float32)
 al=f(a-ah);bl=f(b-bh);p=f(a*b);e=f(f(f(f(ah*bh)-p)+f(ah*bl))+f(al*bh));e=f(e+f(al*bl));t=f(p+c);part=f(t-p);res=f(f(p-f(t-part))+f(c-part));return f(t+f(res+e))
coeffs=[[-2.76076847742355e-16,2.00018790482477e-13,-8.60467152213735e-11,5.12229709037114e-08,1.48572235717979e-05,6.37261928875436e-04,4.89352455891786e-03],[1.19825839466702e-06,1.18534705686654e-04,2.26843463243900e-03,4.89352518554385e-03]]
x=f(-1.5959848165512085);sq=f(x*x);rows=[];results=[]
for cs in coeffs:
 a=b=f(cs[0])
 for c in cs[1:]:
  a,b=fma(a,sq,c),split(b,sq,c)
  rows.append({'fma':float(a),'split':float(b),'exact':bool(a==b),'c':c})
 results.append((a,b))
p,ps=f(results[0][0]*x),f(results[0][1]*x);q,qs=results[1];print(json.dumps({'stages':rows,'numerator':[float(p),float(ps)],'denominator':[float(q),float(qs)],'rounded_division':[float(f(p/q)),float(f(ps/qs))]},indent=2))
print('detailed')
a=f(coeffs[1][0])
for c in coeffs[1][1:]:
 b=sq;c=f(c);ah=(a.view(np.uint32)&np.uint32(0xfffff000)).view(np.float32);bh=(b.view(np.uint32)&np.uint32(0xfffff000)).view(np.float32);al=f(a-ah);bl=f(b-bh);p=f(a*b);e=f(f(f(f(ah*bh)-p)+f(ah*bl))+f(al*bh));e=f(e+f(al*bl));t=f(p+c);part=f(t-p);r=f(f(p-f(t-part))+f(c-part));corr=f(r+e);q=f(t+corr)
 exact=np.longdouble(a)*np.longdouble(b)+np.longdouble(c)
 print(dict(a=float(a),b=float(b),c=float(c),product=float(p),error=float(e),exact_product_error=float(np.longdouble(a)*np.longdouble(b)-np.longdouble(p)),total=float(t),residual=float(r),correction=float(corr),correction_tail=float(np.longdouble(r)+np.longdouble(e)-np.longdouble(corr)),result=float(q),fma=float(fma(a,b,c)),exact=float(exact)))
 a=fma(a,b,c)
