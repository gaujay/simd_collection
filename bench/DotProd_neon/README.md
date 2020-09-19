# Dot Product - ARM

Benchmark configuration:
- OS: Raspberry Pi OS 32-bits
- Compiler: GCC 8.3.0
- Flags: -O2 -DNDEBUG -march=armv7ve+simd -mfpu=neon-vfpv4
- CPU: Raspberry Pi 2 - BCM2836 (L1-D 32K, L2 256K) (overclocked @ 1000MHz)
- Options: no data alignement, max vector size multiple, inner loop '200'


### NEON

![Screenshot_00](../../docs/DotProd/ARM/DotProd_RPi2_Gcc_neon_high.png)
![Screenshot_01](../../docs/DotProd/ARM/DotProd_RPi2_Gcc_neon.png)


### int8 x int8

![Screenshot_04](../../docs/DotProd/ARM/DotProd_RPi2_Gcc_i8.png)


### int16 x int8

![Screenshot_06](../../docs/DotProd/ARM/DotProd_RPi2_Gcc_i16i8.png)


### int16 x int16

![Screenshot_07](../../docs/DotProd/ARM/DotProd_RPi2_Gcc_i16.png)


### int32 x int16

![Screenshot_08](../../docs/DotProd/ARM/DotProd_RPi2_Gcc_i32i16.png)


### int32 x int32

![Screenshot_09](../../docs/DotProd/ARM/DotProd_RPi2_Gcc_i32.png)


### float x float

![Screenshot_10](../../docs/DotProd/ARM/DotProd_RPi2_Gcc_flt.png)
