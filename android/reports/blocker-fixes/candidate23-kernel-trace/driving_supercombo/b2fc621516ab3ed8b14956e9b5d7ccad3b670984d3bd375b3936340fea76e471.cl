__kernel void host_conv(__global float* data0_49152, __global float* data1_16384, __global float* data2_786432) {
  float buf0[1];
  int gidx0 = get_group_id(0); /* 1536 */
  int gidx1 = get_group_id(1); /* 32 */
  *(buf0+0) = 0.0f;
  for (int Lidx1 = 0; Lidx1 < 512; Lidx1++) {
    float val0 = (*(data1_16384+((gidx1<<9)+Lidx1)));
    float val1 = (*(data2_786432+((gidx0<<9)+Lidx1)));
    float alu1 = fma(val0, val1, (*(buf0+0)));
    *(buf0+0) = alu1;
  }
  *(data0_49152+((gidx1*1536)+gidx0)) = (*(buf0+0));
}