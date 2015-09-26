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
    int M=200*200*200*10;
    double *uniform_d=(double *)malloc(sizeof(double)*N1*N2*N3*2); //the output (complex)
    double *x=(double *)malloc(sizeof(double)*M); //the input sample locations
    double *y=(double *)malloc(sizeof(double)*M); //the input sample locations
    double *z=(double *)malloc(sizeof(double)*M); //the input sample locations
    double *nonuniform_d=(double *)malloc(sizeof(double)*M*2); //the input data values (complex)
    double eps=1e-6; //this will determine the spreading kernel size! For gaussian, 1e-3 gives 8, 1e-6 gives 16, I believe
    int K1=50,K2=50,K3=50; //the block size for blocking. You could set these to 10000 to get just a single block.
    int num_threads=4;
	int kernel_type=KERNEL_TYPE_KB;


    printf("Preparing nonuniform locations...\n");
    for (int ii=0; ii<M; ii++) {
        x[ii]=rand()*1.0/RAND_MAX*2*M_PI-M_PI;
        y[ii]=rand()*1.0/RAND_MAX*2*M_PI-M_PI;
        z[ii]=rand()*1.0/RAND_MAX*2*M_PI-M_PI;
    }

    printf("Preparing nonuniform data...\n");
    for (int ii=0; ii<M; ii++) {
        nonuniform_d[ii*2]=rand()*1.0/RAND_MAX * 2 - 1;
        nonuniform_d[ii*2+1]=rand()*1.0/RAND_MAX * 2 - 1;
    }

    printf("Running blocknufft3d_create_plan...\n");
    void *plan=blocknufft3d_create_plan(N1,N2,N3,M,x,y,z,eps,K1,K2,K3,num_threads,kernel_type);

    printf("Running blocknufft3d_run...\n");
    blocknufft3d_run(plan,uniform_d,nonuniform_d);

    printf("Destroying the plan...\n");
    blocknufft3d_destroy_plan(plan);

    printf("Deallocating...\n");
    free(uniform_d);
    free(x);
    free(y);
    free(z);
    free(nonuniform_d);

    printf("Done.\n");

	return 0;
}
