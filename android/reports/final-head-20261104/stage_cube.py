from pathlib import Path
import re,difflib
root=Path.cwd();patch=root/'android/patches/ncnn-explicit-casts.patch';s=patch.read_text();src=next((root/'android/build/native/accelerators/ncnn-patched').glob('ncnn-*'));base=next((root/'android/build/native/accelerators/ncnn-source').glob('ncnn-*'))
fma=re.search(r'\+float fold3_mean_fma\(.*?\n\+\}',s,re.S)[0]
fma='\n'.join(l[1:] for l in fma.splitlines()).replace('fold3_mean_fma','fold3_cube_fma')
helper=fma+'''
float fold3_reference_cube(float x)
{
    precise float squared = x * x;
    precise float error = fold3_cube_fma(x, x, -squared);
    precise float cubed = squared * x;
    precise float tail = fold3_cube_fma(squared, x, -cubed);
    precise float correction = fold3_cube_fma(error, x, tail);
    precise float result = cubed + correction;
    return isinf(cubed) || isnan(cubed) ? cubed : result;
}

'''
for filename in ['binaryop.comp','binaryop_pack4.comp']:
 rel='src/layer/vulkan/shader/'+filename
 old=(src/rel).read_text();new=old.replace('void main()',helper+'void main()',1)
 # This shape is the explicitly mapped original NCHW [1,1280,1,1] cube.
 channels=320 if 'pack4' in filename else 1280
 gate=f'psc(adims) == 3 && psc(aw) == 1 && psc(ah) == 1 && psc(ac) == {channels}'
 call='res = afp(fold3_reference_cube(float(v1)));' if channels==1280 else 'res = afpvec4(fold3_reference_cube(float(v1.x)), fold3_reference_cube(float(v1.y)), fold3_reference_cube(float(v1.z)), fold3_reference_cube(float(v1.w)));'
 needle='            res = square * v1;'
 assert new.count(needle)==1
 new=new.replace(needle,'''            // The original final NCHW singleton cube uses powf; spatial NHWC
            // cubes use the ordered multiply path. Preserve that boundary.
            if ('''+gate+''')
                '''+call+'''
            else res = square * v1;''')
 diff=''.join(difflib.unified_diff((base/rel).read_text().splitlines(True),new.splitlines(True),fromfile='a/'+rel,tofile='b/'+rel))
 start=s.index('--- a/'+rel+'\n');end=s.find('--- a/',start+5);end=len(s) if end<0 else end
 s=s[:start]+diff+s[end:]
patch.write_text(s)
print('Staged compensated final cube')
