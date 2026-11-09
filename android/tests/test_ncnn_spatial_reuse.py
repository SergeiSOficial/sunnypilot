"""Exercise the actual tiled shader body against independent scalar convolution."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class SpatialReuseTest(unittest.TestCase):
  def test_packed_and_scalar_inputs_outputs_and_partial_tiles(self):
    compiler = shutil.which("clang++")
    if compiler is None:
      self.skipTest("Host C++ compiler unavailable")
    patch = (ROOT / "android/patches/ncnn-spatial-weight-reuse.patch").read_text()
    body = "\n".join(line[1:] for line in patch.splitlines()
                     if line.startswith(("+", " ")) and not line.startswith("+++"))
    body = body[body.index("        // Reuse"):body.index("        return;")]
    source = r'''
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <vector>
#define precise
#define psc(x) x
struct vec4 {
  float v[4];
  vec4(float x=0):v{x,x,x,x}{}
  vec4(float a,float b,float c,float d):v{a,b,c,d}{}
  float &operator[](int i){return v[i];}
};
std::vector<float> buffers[4];
vec4 buffer_ld4(int id,int offset){vec4 x;for(int i=0;i<4;i++)x[i]=buffers[id].at(offset*4+i);return x;}
float buffer_ld1(int id,int offset){return buffers[id].at(offset);}
void buffer_st4(int id,int offset,vec4 x){for(int i=0;i<4;i++)buffers[id].at(offset*4+i)=x[i];}
void buffer_st1(int id,int offset,float x){buffers[id].at(offset)=x;}
vec4 fold3_spatial_fma(vec4 a,vec4 b,vec4 c){for(int i=0;i<4;i++)c[i]=std::fma(a[i],b[i],c[i]);return c;}
vec4 fold3_half_stem_fma(vec4 a,vec4 b,vec4 c){return fold3_spatial_fma(a,b,c);}
void tiled(int gx,int gy,int outw,int outsize,int w,int cstep,int outcstep,int num_input,int num_output,int elempack,int out_elempack){
  const int c=num_input/4;
  const int bias_data=0,bottom_blob_data_1=1,bottom_blob_data_4=1,weight_data=2,top_blob_data_1=3,top_blob_data_4=3;
''' + body + r'''
}
int main(){
 uint32_t state=42;
 for(int outw:{1,2,3,5,8,11})for(int outh:{1,3,4})for(int channels:{16,24,32})
 for(int ip:{1,4})for(int op:{1,4}){
   int w=outw*2+1,h=outh*2+1,cstep=w*h,outsize=outw*outh;
   buffers[0].resize(8);buffers[1].resize(channels*cstep);buffers[2].resize(2*(channels/4)*9*16);buffers[3].assign(outsize*8,0);
   for(int b=0;b<3;b++)for(float &x:buffers[b]){state=state*1664525u+1013904223u;x=(int(state%20001)-10000)*0.0001f;}
   for(int gy=0;gy<2;gy++)for(int gx=0;gx<outsize;gx+=4)tiled(gx,gy,outw,outsize,w,cstep,outsize,channels,32,ip,op);
   for(int gy=0;gy<2;gy++)for(int pos=0;pos<outsize;pos++)for(int lane=0;lane<4;lane++){
     float expected=buffers[0][gy*4+lane];
     for(int ky=0;ky<3;ky++)for(int kx=0;kx<3;kx++)for(int ci=0;ci<channels;ci++){
       int spatial=(pos/outw*2+ky)*w+pos%outw*2+kx;
       float input=buffers[1][ip==4?((ci/4)*cstep+spatial)*4+ci%4:ci*cstep+spatial];
       int wi=(gy*(channels/4)*9+(ci/4)*9+ky*3+kx)*16+lane*4+ci%4;
       expected=std::fma(input,buffers[2][wi],expected);
     }
     float actual=buffers[3][op==4?(gy*outsize+pos)*4+lane:(gy*4+lane)*outsize+pos];
     if(std::memcmp(&actual,&expected,4)){std::fprintf(stderr,"spatial reuse mismatch %d %d %d %d %d\n",outw,outh,channels,ip,op);return 1;}
   }
 }
}
'''
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory)
      (path / "spatial.cpp").write_text(source)
      subprocess.run([compiler, "-std=c++17", "-O2", "-ffp-contract=off", str(path / "spatial.cpp"), "-o", str(path / "spatial")], check=True)
      subprocess.run([str(path / "spatial")], check=True)

  def test_patch_is_bound_to_build_and_package(self):
    self.assertIn('"spatial_patch_sha256"', (ROOT / "android/scripts/build_ncnn.py").read_text())
    self.assertIn("'spatial_patch_sha256'", (ROOT / "android/app/build.gradle").read_text())


if __name__ == "__main__":
  unittest.main()
