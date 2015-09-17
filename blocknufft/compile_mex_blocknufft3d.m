disp('To create blocknufft3d.o, you must first run something like: g++ -fopenmp -c blocknufft3d.cpp -fPIC -O3');
mex ./_mcwrap/mcwrap_blocknufft3d.cpp ./blocknufft3d.o ./qute.cpp -output ./blocknufft3d -largeArrayDims -lm -lgomp -lfftw3 -lfftw3_threads
