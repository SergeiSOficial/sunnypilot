__kernel void host_conv(__global float* data0_691200, __global float* data1_1401664, __global float* data2_4608, __global float* data3_32) {
  float buf0[1];
  int gidx0 = get_group_id(0); /* 32 */
  float val0 = (*(data3_32+gidx0));
  int gidx1 = get_group_id(1); /* 180 */
  int gidx2 = get_group_id(2); /* 120 */
  *(buf0+0) = val0;
  int alu1 = (gidx1<<5);
  for (int Lidx1_0_0 = 0; Lidx1_0_0 < 3; Lidx1_0_0++) {
    for (int Lidx1_0_1 = 0; Lidx1_0_1 < 3; Lidx1_0_1++) {
      for (int Lidx1_1 = 0; Lidx1_1 < 16; Lidx1_1++) {
        float val1 = (*(data1_1401664+(alu1+(Lidx1_0_1<<4)+Lidx1_1+(gidx2*11584)+(Lidx1_0_0*5792))));
        float val2 = (*(data2_4608+((Lidx1_0_0*3)+Lidx1_0_1+(Lidx1_1*9)+(gidx0*144))));
        float alu2 = fma(val1, val2, (*(buf0+0)));
        *(buf0+0) = alu2;
      }
    }
  }
  *(data0_691200+(alu1+(gidx2*5760)+gidx0)) = (*(buf0+0));
}