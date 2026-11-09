__kernel void host_conv(__global float* data0_98304, __global float* data1_32768, __global float* data2_196608) {
  float buf0[1];
  int gidx0 = get_group_id(0); /* 768 */
  int gidx1 = get_group_id(1); /* 128 */
  *(buf0+0) = 0.0f;
  for (int Lidx1 = 0; Lidx1 < 256; Lidx1++) {
    float val0 = (*(data1_32768+((gidx1<<8)+Lidx1)));
    float val1 = (*(data2_196608+((gidx0<<8)+Lidx1)));
    float alu1 = fma(val0, val1, (*(buf0+0)));
    *(buf0+0) = alu1;
  }
  *(data0_98304+((gidx1*768)+gidx0)) = (*(buf0+0));
}