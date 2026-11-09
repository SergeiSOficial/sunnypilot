__kernel void host_Tanh(__global float* data0_691200, __global float* data1_691200) {
  int gidx0 = get_group_id(0); /* 21600 */
  int lidx0 = get_local_id(0); /* 32 */
  int alu0 = ((gidx0<<5)+lidx0);
  float val0 = (*(data1_691200+alu0));
  float alu1 = ((val0<-9.0f)?-9.0f:val0);
  float alu2 = -alu1;
  float alu3 = ((alu2<-9.0f)?-9.0f:alu2);
  float alu4 = -alu3;
  float alu5 = alu4 * alu4;
  float alu6 = fma(-2.7607683663038313e-16f, alu5, 2.0001879384549948e-13f);
  float alu7 = fma(1.1982583600911312e-06f, alu5, 0.00011853470641653985f);
  float alu8 = fma(alu6, alu5, -8.604671836165423e-11f);
  float alu9 = fma(alu7, alu5, 0.0022684347350150347f);
  float alu10 = fma(alu8, alu5, 5.122297253024044e-08f);
  float alu11 = fma(alu9, alu5, 0.0048935250379145145f);
  float alu12 = fma(alu10, alu5, 1.4857223504805006e-05f);
  float alu13 = fma(alu12, alu5, 0.0006372619536705315f);
  float alu14 = fma(alu13, alu5, 0.004893524572253227f);
  float alu15 = alu14 * alu4;
  float alu16 = (alu15 / alu11);
  float alu17 = fma(-alu16, alu11, alu15);
  float alu18 = (alu17 / alu11);
  float alu19 = alu16 + alu18;
  float alu20 = (isfinite(alu15) && isfinite(alu11) && isfinite(alu16) && alu11 != 0.0f) ? alu19 : alu16;
  *(data0_691200+alu0) = alu20;
}