#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "omp.h"
#include "blocknufft3d.h"
#include "qute.h"
#include "besseli.h"

int main(int argc, char *argv[])
{
    int N1=200;
    int N2=200;
    int N3=200;
	int M=200*200*200;
    double *uniform_d=(double *)malloc(sizeof(double)*N1*N2*N3*2); //the output (complex)
    double *xyz=(double *)malloc(sizeof(double)*M*3); //the input sample locations
    double *nonuniform_d=(double *)malloc(sizeof(double)*M*2); //the input data values (complex)
    double eps=1e-6; //this will determine the spreading kernel size! For gaussian, 1e-3 gives 8, 1e-6 gives 16, I believe
    int K1=50,K2=50,K3=50; //the block size for blocking. You could set these to 10000 to get just a single block.
    int num_threads=1;
	int kernel_type=KERNEL_TYPE_KB;

    int cc=0;
    for (int aa=0; aa<3; aa++) {
        for (int ii=0; ii<M; ii++) {
            xyz[cc]=rand()*1.0/RAND_MAX*2*M_PI;
            cc++;
        }
    }

    for (int ii=0; ii<M; ii++) {
        nonuniform_d[ii]=rand()*1.0/RAND_MAX * 2 - 1;
    }

    //here is the procedure...
    printf("Running blocknufft3d...\n");
    QTime timer; timer.start();
	blocknufft3d(N1,N2,N3,M,uniform_d,xyz,nonuniform_d,eps,K1,K2,K3,num_threads,kernel_type);
    printf("ELAPSED TIME: %d\n",timer.elapsed());

    free(uniform_d);
    free(xyz);
    free(nonuniform_d);

	return 0;
}
