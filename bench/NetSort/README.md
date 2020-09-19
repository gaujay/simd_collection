# NSort 8

Benchmark configuration:
- OS: Windows 10 64-bits
- Compiler: MinGW 8.1.0 64-bits
- Flags: -O2 -DNDEBUG -march=native -mtune=native
- CPU: i7-10875h (L1-D 32K, L1-I 32K, L2 256K, L3 16M) (turbo mode disabled)
- Options: no data alignement, no early exit, inner loop '100'


### Ordered

![Screenshot_00](../../docs/NSort/NSort8_seq_10875h_MinGW.png)


### Random

![Screenshot_01](../../docs/NSort/NSort8_rnd_10875h_MinGW.png)


### int8

![Screenshot_04](../../docs/NSort/NSort8_i8_10875h_MinGW.png)


### int16

![Screenshot_05](../../docs/NSort/NSort8_i16_10875h_MinGW.png)


### int32

![Screenshot_06](../../docs/NSort/NSort8_i32_10875h_MinGW.png)


### float

![Screenshot_07](../../docs/NSort/NSort8_flt_10875h_MinGW.png)


### double

![Screenshot_08](../../docs/NSort/NSort8_dbl_10875h_MinGW.png)
