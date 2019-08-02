This is a work-related project that attempts to beat MATLAB's filter command when dealing with complex numbers. We have two different versions, the first is a vanilla C++ implementation and the second is an AVX version. The results are shown here

![Screenshot](/test.png)

As we can see we have a ~6.5 times speedup (02-Aug-2019). There is some room for improvement still (manual loop unrolling maybe). 
