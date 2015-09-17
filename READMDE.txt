nufft_comparison

Comparing the speeds between various nufft implementations

Compilation is not quite straightforward at this point, but here's what you can try:

Create the mex for the fortran version
> cd nufft-1.33-mcwrap
> compile_mex_nufft

Create the mex for the C++ version - a bit more complicated
Not having an easy time using openmp in mex... so you must first creat a .o file:
> cd blocknufft
> g++ -fopenmp -c blocknufft3d.cpp -fPIC -O3

That's for Linux... need to do equivalent for other OS

Then in MATLAB
> cd blocknufft
> compile_mex_blocknufft3d

Now take a look at do_compare_01.m

There are various options to do various things. Sorry, no more details
at this point.

Fun FESSLER at your own risk -- uses up a lot of RAM, depending on size of data.


