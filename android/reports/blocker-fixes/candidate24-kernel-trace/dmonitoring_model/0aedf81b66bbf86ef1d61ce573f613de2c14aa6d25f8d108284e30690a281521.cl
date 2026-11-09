__kernel void host_conv(__global float* data0_129600, __global float* data1_182528, __global float* data2_27648, __global float* data3_96) {
  float buf0[1];
  int gidx0 = get_group_id(0); /* 3 */
  int lidx0 = get_local_id(0); /* 32 */
  int alu0 = (gidx0<<5);
  float val0 = (*(data3_96+(alu0+lidx0)));
  int gidx1 = get_group_id(1); /* 45 */
  int gidx2 = get_group_id(2); /* 30 */
  *(buf0+0) = val0;
  for (int Lidx1_0_0 = 0; Lidx1_0_0 < 3; Lidx1_0_0++) {
    for (int Lidx1_0_1 = 0; Lidx1_0_1 < 3; Lidx1_0_1++) {
      for (int Lidx1_1 = 0; Lidx1_1 < 32; Lidx1_1++) {
        float val1 = (*(data1_182528+((gidx1<<6)+(Lidx1_0_1<<5)+Lidx1_1+(gidx2*5888)+(Lidx1_0_0*2944))));
        float val2 = (*(data2_27648+((Lidx1_0_0*3)+Lidx1_0_1+(Lidx1_1*9)+(gidx0*9216)+(lidx0*288))));
        float alu2 = fma(val1, val2, (*(buf0+0)));
        *(buf0+0) = alu2;
      }
    }
  }
  *(data0_129600+((gidx1*96)+(gidx2*4320)+alu0+lidx0)) = (*(buf0+0));
}