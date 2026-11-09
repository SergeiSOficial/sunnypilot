"""Stage common FMA midpoint repair; do not mutate artifacts under test."""
from pathlib import Path
import re
root=Path(__file__).resolve().parents[3]
s=(root/'android/patches/ncnn-explicit-casts.patch').read_text()
pattern=r'(\+(?:vec4|float) fold3_\w+_fma\([^\n]+\)\n\+\{.*?)(\+    return total \+ \(residual \+ error\);)(\n\+\})'
def repair(m):
 v=m[1].startswith('+vec4 ')
 t='vec4' if v else 'float'
 result=f'''+    // Preserve a nonzero correction tail when the final sum is a midpoint.
+    precise {t} correction = residual + error;
+    precise {t} correction_part = correction - residual;
+    precise {t} tail = (residual - (correction - correction_part)) + (error - correction_part);
+    precise {t} rounded = total + correction;
+    precise {t} rounded_part = rounded - total;
+    precise {t} remainder = (total - (rounded - rounded_part)) + (correction - rounded_part);
'''
 idx='[lane]' if v else ''
 if v: result+='+    for (int lane = 0; lane < 4; lane++)\n+    {\n'
 indent='+        ' if v else '+    '
 result+=f'''{indent}if (tail{idx} != 0.f && remainder{idx} != 0.f && !isinf(rounded{idx}) && !isnan(rounded{idx}))
{indent}{{
{indent}    uint bits = floatBitsToUint(rounded{idx});
{indent}    bool upwards = remainder{idx} > 0.f;
{indent}    uint adjacent_bits = (upwards == (rounded{idx} >= 0.f)) ? bits + 1u : bits - 1u;
{indent}    float adjacent = uintBitsToFloat(adjacent_bits);
{indent}    precise float half_gap = abs(adjacent - rounded{idx}) * 0.5f;
{indent}    if (abs(remainder{idx}) == half_gap && ((tail{idx} > 0.f) == upwards)) rounded{idx} = adjacent;
{indent}}}
'''
 if v:result+='+    }\n'
 result+='+    return rounded;'
 return m[1]+result+m[3]
s,n=re.subn(pattern,repair,s,flags=re.S)
assert n==7,n
lines=s.splitlines();out=[];i=0;delta=0
while i<len(lines):
 l=lines[i]
 if l.startswith('--- a/'):delta=0
 if l.startswith('@@'):
  h=re.match(r'@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@(.*)',l);j=i+1
  while j<len(lines) and not lines[j].startswith(('@@','--- a/')):j+=1
  body=lines[i+1:j];old=sum(x.startswith((' ','-')) for x in body);new=sum(x.startswith((' ','+')) for x in body)
  out.append(f'@@ -{h[1]},{old} +{int(h[1])+delta},{new} @@{h[2]}');out.extend(body);delta+=new-old;i=j;continue
 out.append(l);i+=1
p=Path(__file__).with_name('all-fma-midpoints.patch');p.write_text('\n'.join(out)+'\n');print(n,p)
