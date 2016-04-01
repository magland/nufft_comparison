#!/bin/bash

g++ -c -lfftw3 -lfftw3_omp -lm -fopenmp besseli.cpp block3dspreader.cpp blocknufft3d_c.cpp blocknufft3d.cpp qute.cpp

