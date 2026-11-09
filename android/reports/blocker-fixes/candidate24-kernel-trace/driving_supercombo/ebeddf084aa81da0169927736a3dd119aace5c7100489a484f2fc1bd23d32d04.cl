__kernel void host_conv(__global float* data0_16384, __global float* data1_49152, __global float* data2_786432) {
  float buf0[1];
  int gidx0 = get_group_id(0); /* 16 */
  int gidx1 = get_group_id(1); /* 32 */
  int lidx0 = get_local_id(0); /* 32 */
  *(buf0+0) = 0.0f;
  for (int Lidx1 = 0; Lidx1 < 1536; Lidx1++) {
    float val0 = (*(data1_49152+((gidx1*1536)+Lidx1)));
    float val1 = (*(data2_786432+((gidx0*49152)+(lidx0*1536)+Lidx1)));
    float alu1 = fma(val0, val1, (*(buf0+0)));
    *(buf0+0) = alu1;
  }
  *(data0_16384+((gidx0<<5)+(gidx1<<9)+lidx0)) = (*(buf0+0));
}