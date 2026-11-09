__kernel void host_conv(__global float* data0_86400, __global float* data1_129600, __global float* data2_6144) {
  float buf0[1];
  int gidx0 = get_group_id(0); /* 64 */
  int gidx1 = get_group_id(1); /* 1350 */
  *(buf0+0) = 0.0f;
  for (int Lidx1 = 0; Lidx1 < 96; Lidx1++) {
    float val0 = (*(data1_129600+((gidx1*96)+Lidx1)));
    float val1 = (*(data2_6144+((gidx0*96)+Lidx1)));
    float alu1 = fma(val0, val1, (*(buf0+0)));
    *(buf0+0) = alu1;
  }
  *(data0_86400+((gidx1<<6)+gidx0)) = (*(buf0+0));
}