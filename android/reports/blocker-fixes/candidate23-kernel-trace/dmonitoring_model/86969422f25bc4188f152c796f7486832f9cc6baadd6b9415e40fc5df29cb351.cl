__kernel void host_conv(__global float* data0_259200, __global float* data1_86400, __global float* data2_12288) {
  float buf0[1];
  int gidx0 = get_group_id(0); /* 192 */
  int gidx1 = get_group_id(1); /* 1350 */
  *(buf0+0) = 0.0f;
  for (int Lidx1 = 0; Lidx1 < 64; Lidx1++) {
    float val0 = (*(data1_86400+((gidx1<<6)+Lidx1)));
    float val1 = (*(data2_12288+((gidx0<<6)+Lidx1)));
    float alu1 = fma(val0, val1, (*(buf0+0)));
    *(buf0+0) = alu1;
  }
  *(data0_259200+((gidx1*192)+gidx0)) = (*(buf0+0));
}