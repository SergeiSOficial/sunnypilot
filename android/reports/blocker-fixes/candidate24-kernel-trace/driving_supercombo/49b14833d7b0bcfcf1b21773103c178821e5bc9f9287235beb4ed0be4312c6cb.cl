__kernel void host_conv(__global float* data0_524288, __global float* data1_804960, __global float* data2_13824, __global float* data3_64) {
  float buf0[1];
  int gidx0 = get_group_id(0); /* 2 */
  int lidx0 = get_local_id(0); /* 32 */
  int alu0 = (gidx0<<5);
  float val0 = (*(data3_64+(alu0+lidx0)));
  int gidx1 = get_group_id(1); /* 128 */
  int gidx2 = get_group_id(2); /* 64 */
  *(buf0+0) = val0;
  for (int Lidx1_0_0 = 0; Lidx1_0_0 < 3; Lidx1_0_0++) {
    for (int Lidx1_0_1 = 0; Lidx1_0_1 < 3; Lidx1_0_1++) {
      for (int Lidx1_1 = 0; Lidx1_1 < 24; Lidx1_1++) {
        float val1 = (*(data1_804960+((gidx1*48)+(Lidx1_0_1*24)+Lidx1_1+(gidx2*12384)+(Lidx1_0_0*6192))));
        float val2 = (*(data2_13824+((Lidx1_0_0*3)+Lidx1_0_1+(Lidx1_1*9)+(gidx0*6912)+(lidx0*216))));
        float alu2 = fma(val1, val2, (*(buf0+0)));
        *(buf0+0) = alu2;
      }
    }
  }
  *(data0_524288+((gidx1<<6)+(gidx2<<13)+alu0+lidx0)) = (*(buf0+0));
}