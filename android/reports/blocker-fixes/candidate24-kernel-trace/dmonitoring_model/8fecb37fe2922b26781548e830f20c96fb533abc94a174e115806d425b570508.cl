__kernel void host_conv(__global float* data0_172800, __global float* data1_710528, __global float* data2_9216, __global float* data3_32) {
  float buf0[1];
  int lidx0 = get_local_id(0); /* 32 */
  float val0 = (*(data3_32+lidx0));
  int gidx0 = get_group_id(0); /* 90 */
  int gidx1 = get_group_id(1); /* 60 */
  *(buf0+0) = val0;
  for (int Lidx1_0_0 = 0; Lidx1_0_0 < 3; Lidx1_0_0++) {
    for (int Lidx1_0_1 = 0; Lidx1_0_1 < 3; Lidx1_0_1++) {
      for (int Lidx1_1 = 0; Lidx1_1 < 32; Lidx1_1++) {
        float val1 = (*(data1_710528+((gidx0<<6)+(Lidx1_0_1<<5)+Lidx1_1+(gidx1*11648)+(Lidx1_0_0*5824))));
        float val2 = (*(data2_9216+((Lidx1_0_0*3)+Lidx1_0_1+(Lidx1_1*9)+(lidx0*288))));
        float alu1 = fma(val1, val2, (*(buf0+0)));
        *(buf0+0) = alu1;
      }
    }
  }
  *(data0_172800+((gidx0<<5)+(gidx1*2880)+lidx0)) = (*(buf0+0));
}