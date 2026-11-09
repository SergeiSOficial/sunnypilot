__kernel void host_conv(__global float* data0_198720, __global float* data1_33120, __global float* data2_55296) {
  float buf0[1];
  int gidx0 = get_group_id(0); /* 18 */
  int gidx1 = get_group_id(1); /* 345 */
  int lidx0 = get_local_id(0); /* 32 */
  *(buf0+0) = 0.0f;
  for (int Lidx1 = 0; Lidx1 < 96; Lidx1++) {
    float val0 = (*(data1_33120+((gidx1*96)+Lidx1)));
    float val1 = (*(data2_55296+((gidx0*3072)+(lidx0*96)+Lidx1)));
    float alu1 = fma(val0, val1, (*(buf0+0)));
    *(buf0+0) = alu1;
  }
  *(data0_198720+((gidx0<<5)+(gidx1*576)+lidx0)) = (*(buf0+0));
}