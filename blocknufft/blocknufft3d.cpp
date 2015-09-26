#include "blocknufft3d.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "qute.h"
#include "omp.h"
#include "fftw3.h"
#include "besseli.h"
#include "unistd.h"
#include "block3dspreader.h"

//#define NUFFT_ANALYTIC_CORRECTION
//#define USE_COMPLEX_CORRECTION

//Redundant?
struct BlockSpread3DOptions {
	int N1o,N2o,N3o; //size of grid
	int M; //number of non-uniform points
};

// This is the data needed for the spreading
struct BlockSpread3DData {
	int M; // Number of non-uniform points
	int N1o,N2o,N3o; // Oversampled dimensions
	double *x,*y,*z; // non-uniform locations
	double *nonuniform_d; // non-uniform data
	double *uniform_d; // uniform data
};

// A couple implementation routines for nufft
bool do_fft_3d(int N1,int N2,int N3,double *out,double *in,int num_threads=1);
void do_fix_3d(int N1,int N2,int N3,int M,KernelInfo KK1,KernelInfo KK2,KernelInfo KK3,double *out,double *out_oversamp_hat);

// These are the implementation routines for spreading
void define_block_ids_and_location_codes(BlockSpread3DData &D);
void compute_nonuniform_block_counts(BlockSpread3DData &D);
void compute_sizes_and_block_indices(BlockSpread3DData &D);
void set_working_nonuniform_data(BlockSpread3DData &D);
void set_uniform_data(BlockSpread3DData &D);

//set a lookup table for fast Gaussian spreading (which, btw, is not as good as KB)
#define MAX_LOOKUP_EXP 200
double s_lookup_exp1[MAX_LOOKUP_EXP];
double s_lookup_exp2[MAX_LOOKUP_EXP];
double s_lookup_exp3[MAX_LOOKUP_EXP];
void setup_lookup_exp1(double tau) {
    for (int i=0; i<MAX_LOOKUP_EXP; i++) {
        s_lookup_exp1[i]=exp(-i*i*tau);
    }
}
void setup_lookup_exp2(double tau) {
    for (int i=0; i<MAX_LOOKUP_EXP; i++) {
        s_lookup_exp2[i]=exp(-i*i*tau);
    }
}
void setup_lookup_exp3(double tau) {
    for (int i=0; i<MAX_LOOKUP_EXP; i++) {
        s_lookup_exp3[i]=exp(-i*i*tau);
    }
}


//Creates a plan and returns a pointer to a Block3DSpreader object
void *blocknufft3d_create_plan(BlockNufft3DOptions opts,double *x_in,double *y_in,double *z_in) {
    QTime blocknufft3d_create_plan_timer; blocknufft3d_create_plan_timer.start();
    QTime timer0; timer0.start();

    printf ("\nStarting precompute_blocknufft3d.\n");

    int initialization_time=timer0.elapsed();
    printf ("  Initialization: %d ms\n",(int)initialization_time);

    //check the inputs
    for (int i=0; i<opts.M; i++) {
        if ((x_in[i]<-M_PI)||(x_in[i]>=M_PI)||(y_in[i]<-M_PI)||(y_in[i]>=M_PI)||(z_in[i]<-M_PI)||(z_in[i]>=M_PI)) {
            printf("PROBLEM preparing spreader: input locations are out of range (should be >=-pi and <pi\n");
            return 0;
        }
    }

    KernelInfo KK1,KK2,KK3;

    if (opts.kernel_type==KERNEL_TYPE_KB) {
        KK1.kernel_type=KERNEL_TYPE_KB;
        KK2.kernel_type=KERNEL_TYPE_KB;
        KK3.kernel_type=KERNEL_TYPE_KB;

        KK1.oversamp=2;
        KK2.oversamp=2;
        KK3.oversamp=2;
        //int nspread=10; double fac1=0.90,fac2=1.47;
        int nspread=12; double fac1=1,fac2=1;
        if (opts.eps>=1e-2) {
            nspread=4; fac1=0.75; fac2=1.71;
        }
        else if (opts.eps>=1e-4) {
            nspread=6; fac1=0.83; fac2=1.56;
        }
        else if (opts.eps>=1e-6) {
            nspread=8; fac1=0.89; fac2=1.45;
        }
        else if (opts.eps>=1e-8) {
            nspread=10; fac1=0.90; fac2=1.47;
        }
        else if (opts.eps>=1e-10) {
            nspread=12; fac1=0.92; fac2=1.51;
        }
        else if (opts.eps>=1e-12) {
            nspread=14; fac1=0.94; fac2=1.48;
        }
        else {
            nspread=16; fac1=0.94; fac2=1.46;
        }

        {
            KK1.nspread=nspread;
            KK1.W=KK1.nspread*fac1;
            double tmp0=KK1.W*KK1.W/4-0.8;
            if (tmp0<0) tmp0=0; //fix this?
            KK1.beta=M_PI*sqrt(tmp0)*fac2;
        }

        {
            KK2.nspread=nspread;
            KK2.W=KK2.nspread*fac1;
            double tmp0=KK2.W*KK2.W/4-0.8;
            if (tmp0<0) tmp0=0; //fix this?
            KK2.beta=M_PI*sqrt(tmp0)*fac2;
        }

        {
            KK3.nspread=nspread;
            KK3.W=KK3.nspread*fac1;
            double tmp0=KK3.W*KK3.W/4-0.8;
            if (tmp0<0) tmp0=0; //fix this?
            KK3.beta=M_PI*sqrt(tmp0)*fac2;
        }
    }
    else if (opts.kernel_type==KERNEL_TYPE_GAUSSIAN) {
        double eps=opts.eps * 10; //note: jfm multiplied by 100 here!
        double oversamp=2; if (eps<= 1e-11) oversamp=3;

        KK1.oversamp=oversamp;
        KK2.oversamp=oversamp;
        KK3.oversamp=oversamp;

        int nspread=(int)(-log(eps)/(M_PI*(oversamp-1)/(oversamp-.5)) + .5) + 1; //the plus one was added -- different from docs -- aha!
        nspread=nspread*2; //we need to multiply by 2, because I consider nspread as the diameter
        double lambda=oversamp*oversamp * nspread/2 / (oversamp*(oversamp-.5));
        double tau=M_PI/lambda;
        printf ("Using oversamp=%g, nspread=%d, tau=%g\n",oversamp,nspread,tau);

        KK1.kernel_type=KERNEL_TYPE_GAUSSIAN;
        KK2.kernel_type=KERNEL_TYPE_GAUSSIAN;
        KK3.kernel_type=KERNEL_TYPE_GAUSSIAN;


        KK1.tau=tau;
        KK1.nspread=nspread;
        setup_lookup_exp1(KK1.tau);
        KK1.lookup_exp=s_lookup_exp1;

        KK2.tau=tau;
        KK2.nspread=nspread;
        setup_lookup_exp2(KK2.tau);
        KK2.lookup_exp=s_lookup_exp2;

        KK3.tau=tau;
        KK3.nspread=nspread;
        setup_lookup_exp3(KK3.tau);
        KK3.lookup_exp=s_lookup_exp3;
    }
    else {
        printf ("Unknown kernel type: %d\n",opts.kernel_type);
        return 0;
    }

    Block3DSpreader *SS=new Block3DSpreader;
    SS->setKernelInfo(KK1,KK2,KK3);
    SS->setN(opts.N1,opts.N2,opts.N3);
    SS->setM(opts.M);

    omp_set_num_threads(opts.num_threads);

    int N1o=(int)(opts.N1*KK1.oversamp); int N2o=(int)(opts.N2*KK2.oversamp); int N3o=(int)(opts.N3*KK3.oversamp);

    double *x=(double *) malloc(sizeof(double)*opts.M);
    double *y=(double *) malloc(sizeof(double)*opts.M);
    double *z=(double *) malloc(sizeof(double)*opts.M);

    //The input locations are between -pi and pi.... we change them to be between 0 and N1o/N2o/N3o
    //But the 0 frequency needs to correspond to 0!
    double factor_x=N1o/(2*M_PI);
    double factor_y=N2o/(2*M_PI);
    double factor_z=N3o/(2*M_PI);
    for (int ii=0; ii<opts.M; ii++) {
        x[ii]=(x_in[ii]>=0) ? x_in[ii]*factor_x : (x_in[ii]+2*M_PI)*factor_x;
        y[ii]=(y_in[ii]>=0) ? y_in[ii]*factor_y : (y_in[ii]+2*M_PI)*factor_y;
        z[ii]=(z_in[ii]>=0) ? z_in[ii]*factor_z : (z_in[ii]+2*M_PI)*factor_z;
    }

    timer0.start();
    //create blocks
    int num_blocks_x=ceil(N1o*1.0/opts.K1);
    int num_blocks_y=ceil(N2o*1.0/opts.K2);
    int num_blocks_z=ceil(N3o*1.0/opts.K3);
    //NOTE: for the combining step (below) we use a checkerboard-style pattern to avoid conflicts in parallelization
    //      so we need to make sure that the number of blocks in each direction is either 1 or is even
    if ((num_blocks_x>1)&&(num_blocks_x%2==1)) num_blocks_x++;
    if ((num_blocks_y>1)&&(num_blocks_y%2==1)) num_blocks_y++;
    if ((num_blocks_z>1)&&(num_blocks_z%2==1)) num_blocks_z++;
    //Now update the block sizes
    opts.K1=ceil(N1o*1.0/num_blocks_x);
    opts.K2=ceil(N2o*1.0/num_blocks_y);
    opts.K3=ceil(N3o*1.0/num_blocks_z);
    {
        int bb=0;
        for (int i3=0; i3<num_blocks_z; i3++) {
            for (int i2=0; i2<num_blocks_y; i2++) {
                for (int i1=0; i1<num_blocks_x; i1++) {
                    BlockData *BD=new BlockData;
                    BD->x_block_index=i1; //we'll need this later for the checkboard-style combining step
                    BD->y_block_index=i2;
                    BD->z_block_index=i3;
                    BD->xmin=i1*opts.K1; BD->xmax=fmin((i1+1)*opts.K1-1,N1o-1); //set the min and max indices for each block
                    BD->ymin=i2*opts.K2; BD->ymax=fmin((i2+1)*opts.K2-1,N2o-1);
                    BD->zmin=i3*opts.K3; BD->zmax=fmin((i3+1)*opts.K3-1,N3o-1);
                    if (i1+1==num_blocks_x) BD->xmax=N1o-1; //make sure the final block goes all the way to the end
                    if (i2+1==num_blocks_y) BD->xmax=N2o-1;
                    if (i3+1==num_blocks_z) BD->xmax=N3o-1;
                    BD->N1o=BD->xmax-BD->xmin+1+KK1.nspread; //the block size
                    BD->N2o=BD->ymax-BD->ymin+1+KK2.nspread;
                    BD->N3o=BD->zmax-BD->zmin+1+KK3.nspread;
                    BD->M=0; //we'll tally this up later
                    SS->addBlock(BD); //add this block to the spreader
                    bb++;
                }
            }
        }
    }

    //compute block counts
    for (int ii=0; ii<opts.M; ii++) {
        int i1=((int)(x[ii]))/opts.K1;
        int i2=((int)(y[ii]))/opts.K2;
        int i3=((int)(z[ii]))/opts.K3;
        if ((i1>=num_blocks_x)||(i2>=num_blocks_y)||(i3>=num_blocks_z)) {
            printf ("Unexpected problem computing block counts!!!\n");
        }
        else {
            SS->block(i1+num_blocks_x*i2+num_blocks_x*num_blocks_y*i3)->M++;
        }
    }

    //allocate block data
    printf ("Allocating block data...\n"); timer0.start();
    for (int bb=0; bb<SS->blockCount(); bb++) {
        BlockData *BD=SS->block(bb);
        BD->x=(double *)malloc(sizeof(double)*BD->M);
        BD->y=(double *)malloc(sizeof(double)*BD->M);
        BD->z=(double *)malloc(sizeof(double)*BD->M);
        BD->nonuniform_d=(double *)malloc(sizeof(double)*BD->M*2);
        BD->uniform_d=(double *)malloc(sizeof(double)*BD->N1o*BD->N2o*BD->N3o*2);
        BD->nonuniform_indices=(int *)malloc(sizeof(int)*BD->M);
        BD->jj=0;
    }

    //set block input data
    for (int ii=0; ii<opts.M; ii++) {
        int i1=((int)(x[ii]))/opts.K1;
        int i2=((int)(y[ii]))/opts.K2;
        int i3=((int)(z[ii]))/opts.K3;
        if ((i1>=num_blocks_x)||(i2>=num_blocks_y)||(i3>=num_blocks_z)) {
            printf ("Unexpected problem setting block input data!!!\n");
        }
        else {
            BlockData *BD=SS->block(i1+num_blocks_x*i2+num_blocks_x*num_blocks_y*i3);
            int jj=BD->jj;
            //BD->x[jj]=x[ii]-BD->xmin+KK1.nspread/2;
            //BD->y[jj]=y[ii]-BD->ymin+KK2.nspread/2;
            //BD->z[jj]=z[ii]-BD->zmin+KK3.nspread/2;
            //BD->nonuniform_d[jj*2]=d[ii*2];
            //BD->nonuniform_d[jj*2+1]=d[ii*2+1];
            BD->nonuniform_indices[jj]=ii;
            BD->jj++;
        }
    }
    for (int bb=0; bb<SS->blockCount(); bb++) {
        BlockData *BD=SS->block(bb);
        for (int mm=0; mm<BD->M; mm++) {
            int ii=BD->nonuniform_indices[mm];
            BD->x[mm]=x[ii]-BD->xmin+KK1.nspread/2;
            BD->y[mm]=y[ii]-BD->ymin+KK2.nspread/2;
            BD->z[mm]=z[ii]-BD->zmin+KK3.nspread/2;
        }
    }

    double prepare_blocks_time=timer0.elapsed();
    printf ("  Prepare blocks: %d ms\n",(int)prepare_blocks_time);

    timer0.start();
    if (opts.num_threads>1) SS->setParallel(PARALLEL_OPENMP,opts.num_threads);
    else SS->setParallel(PARALLEL_NONE,1);
    SS->precompute();
    double precompute_time=timer0.elapsed();
    printf ("  Precompute: %d ms\n",(int)precompute_time);

    free(x); free(y); free(z);

    printf("ELAPSED TIME for blocknufft3d_create_plan: %.3f seconds\n",blocknufft3d_create_plan_timer.elapsed()*1.0/1000);

    return SS;
}

//Anoter interface for creating the plan
void *blocknufft3d_create_plan(int N1,int N2,int N3,int M,double *x,double *y,double *z,double eps,int K1,int K2,int K3,int num_threads,int kernel_type) {
    BlockNufft3DOptions opts;
    opts.N1=N1; opts.N2=N2; opts.N3=N3;
    opts.M=M;
    opts.K1=K1; opts.K2=K2; opts.K3=K3;
    opts.eps=eps;
    opts.kernel_type=kernel_type;
    opts.num_threads=num_threads;

    return blocknufft3d_create_plan(opts,x,y,z);
}

// Here's the nufft run!
bool blocknufft3d_run(void *plan,double *out,double *d) {

    if (!plan) {
        printf("Problem running nufft... plan is null.\n");
        return false;
    }

    Block3DSpreader *SS=(Block3DSpreader *)plan;
    QTime blocknufft3d_timer; blocknufft3d_timer.start();
    omp_set_num_threads(SS->numThreads());
    QTime timer0;

    ////////////////////////////////////////////////////////////////////
    timer0.start();
    int N1o=(int)(SS->N1()*SS->KK1().oversamp); int N2o=(int)(SS->N2()*SS->KK2().oversamp); int N3o=(int)(SS->N3()*SS->KK3().oversamp);

    double *out_oversamp=(double *)malloc(sizeof(double)*N1o*N2o*N3o*2);
    double *out_oversamp_hat=(double *)malloc(sizeof(double)*N1o*N2o*N3o*2);

    for (int bb=0; bb<SS->blockCount(); bb++) {
        BlockData *BD=SS->block(bb);
        for (int mm=0; mm<BD->M; mm++) {
            int ii=BD->nonuniform_indices[mm];
            BD->nonuniform_d[mm*2]=d[ii*2];
            BD->nonuniform_d[mm*2+1]=d[ii*2+1];
        }
    }
    double setup_time=timer0.elapsed();
    printf ("  Setting up data: %d ms\n",(int)setup_time);

    ////////////////////////////////////////////////////////////////////
    timer0.start();
    SS->run();
    double spreading_operation_count=0;
    for (int j=0; j<SS->blockCount(); j++) {
        BlockData *BD=SS->block(j);
        spreading_operation_count+=BD->M*SS->KK1().nspread*SS->KK2().nspread*SS->KK3().nspread*4;
    }
    double spreading_time=timer0.elapsed();
    printf ("  Spreading: %d ms (%g megaflops)\n",(int)spreading_time,spreading_operation_count*1.0/spreading_time*1000/1e6);

    ////////////////////////////////////////////////////////////////////
    timer0.start();
	int N1oN2oN3o=N1o*N2o*N3o;
	for (int ii=0; ii<N1oN2oN3o; ii++) {
		out_oversamp[ii*2]=0;
		out_oversamp[ii*2+1]=0;
	}

    int nspread1=SS->KK1().nspread; int nspread2=SS->KK2().nspread; int nspread3=SS->KK3().nspread;

    /*
     * Now we need to combine the output data from the blocks.
     * And we'd like to parallelize this step. However, due to overlaps, we cannot just let every
     * block have at it simultaneously. So we need to use a checkerboard pattern and do 8 passes!
     * This is why we NEED to make sure we have an even number of blocks in each direction (or 1 is okay)!!
     * This is guaranteed/handled above
     */
    QTime combining_timer; combining_timer.start(); int combining_operation_count=0;
    for (int bb_z_parity=0; bb_z_parity<2; bb_z_parity++)
    for (int bb_y_parity=0; bb_y_parity<2; bb_y_parity++)
    for (int bb_x_parity=0; bb_x_parity<2; bb_x_parity++) {
        #pragma omp parallel
        {
            int local_operation_count=0;
            #pragma omp for
            for (int bb=0; bb<SS->blockCount(); bb++) {
                BlockData *BD=SS->block(bb);
                if (((BD->x_block_index%2)==bb_x_parity)&&((BD->y_block_index%2)==bb_y_parity)&&((BD->z_block_index%2)==bb_z_parity)) {
                    if ((0<=BD->xmin-nspread1/2)&&(BD->xmax+nspread1/2<N1o)&&(0<=BD->ymin-nspread2/2)&&(BD->ymax+nspread2/2<N2o)&&(0<=BD->zmin-nspread3/2)&&(BD->zmax+nspread3/2<N3o)) {
                        //in this case we don't need to worry about modulos (wrapping)
                        int jjj=0;
                        int min1=BD->xmin-nspread1/2; int max1=min1+BD->N1o;
                        int min2=BD->ymin-nspread2/2; int max2=min2+BD->N2o;
                        int min3=BD->zmin-nspread3/2; int max3=min3+BD->N3o;
                        for (int j3=min3; j3<max3; j3++) {
                            int kkk3=j3*N1o*N2o;
                            for (int j2=min2; j2<max2; j2++) {
                                int kkk2=kkk3+j2*N1o;
                                for (int j1=min1+kkk2; j1<max1+kkk2; j1++) { //we absorb the +kkk2 into the loop, not sure if it helps
                                    out_oversamp[j1*2]+=BD->uniform_d[jjj*2];
                                    out_oversamp[j1*2+1]+=BD->uniform_d[jjj*2+1];
                                    jjj++;
                                }
                                local_operation_count+=BD->N1o*2;
                            }
                        }
                    }
                    else {
                        //in this case we need to worry about modulos (wrapping)
                        int jjj=0;
                        int min1=BD->xmin-nspread1/2; int max1=min1+BD->N1o;
                        int min2=BD->ymin-nspread2/2; int max2=min2+BD->N2o;
                        int min3=BD->zmin-nspread3/2; int max3=min3+BD->N3o;
                        for (int j3=min3; j3<max3; j3++) {
                            int kkk3=((j3+N3o)%N3o)*N1o*N2o;
                            for (int j2=min2; j2<max2; j2++) {
                                int kkk2=kkk3+((j2+N2o)%N2o)*N1o;
                                for (int j1=min1; j1<max1; j1++) {
                                    int k1=((j1+N1o)%N1o)+kkk2;
                                    out_oversamp[k1*2]+=BD->uniform_d[jjj*2];
                                    out_oversamp[k1*2+1]+=BD->uniform_d[jjj*2+1];
                                    jjj++;
                                }
                                local_operation_count+=BD->N1o*2;
                            }
                        }
                    }
                }
            }
            combining_operation_count+=local_operation_count;
        }
    }
    double combining_time=timer0.elapsed();
    printf ("  Combining: %d ms (%g megaflops)\n",(int)combining_time,combining_operation_count*1.0/combining_time*1000/1e6);

    ////////////////////////////////////////////////////////////////////
    timer0.start();
    //fft
    if (!do_fft_3d(N1o,N2o,N3o,out_oversamp_hat,out_oversamp,SS->numThreads())) {
        printf ("problem in do_fft_3d\n");
		free(out_oversamp);
		free(out_oversamp_hat);
		return false;
	}
	double fft_time=timer0.elapsed();
    printf ("  FFT: %d ms\n",(int)fft_time);

    ////////////////////////////////////////////////////////////////////
    timer0.start();
    //correction
    do_fix_3d(SS->N1(),SS->N2(),SS->N3(),SS->M(),SS->KK1(),SS->KK2(),SS->KK3(),out,out_oversamp_hat);
    double correction_time=timer0.elapsed();
    printf ("  Correction: %d ms\n",(int)correction_time);

    ////////////////////////////////////////////////////////////////////
    //finalize
    timer0.start();

	free(out_oversamp_hat);
	free(out_oversamp);
    double finalize_time=timer0.elapsed();
    printf ("  Finalize: %d ms\n",(int)finalize_time);

    double total_time=blocknufft3d_timer.elapsed();

    printf ("   %.3f spreading, %.3f fft, %.3f other\n",spreading_time/1000,fft_time/1000,(total_time-spreading_time-fft_time)/1000);
    printf ("   %.1f%% spreading, %.1f%% fft, %.1f%% other\n",spreading_time/total_time*100,fft_time/total_time*100,(total_time-spreading_time-fft_time)/total_time*100);
    printf ("ELAPSED TIME for blocknufft3d_run: %.3f seconds\n",total_time/1000);

    printf ("done with blocknufft3d.\n");

	return true;
}

//Be sure to destroy the plan to free up memory
void blocknufft3d_destroy_plan(void *plan) {
    if (!plan) return;
    Block3DSpreader *SS=(Block3DSpreader *)plan;
    delete SS;
}

// This is the mcwrap interface that does both create_plan and run from MATLAB
void blocknufft3d(int N1,int N2,int N3,int M,double *uniform_d,double *xyz,double *nonuniform_d,double eps,int K1,int K2,int K3,int num_threads,int kernel_type) {
	BlockNufft3DOptions opts;
	opts.eps=eps;
	opts.K1=K1; opts.K2=K2; opts.K3=K3;
	opts.N1=N1; opts.N2=N2; opts.N3=N3;
	opts.M=M;
	opts.num_threads=num_threads;
	opts.kernel_type=kernel_type;

	double *x=&xyz[0];
	double *y=&xyz[M];
	double *z=&xyz[2*M];
    void *plan=blocknufft3d_create_plan(opts,x,y,z);
    blocknufft3d_run(plan,uniform_d,nonuniform_d);
    blocknufft3d_destroy_plan(plan);
}

/////////////////////////////////////////////
/////////////////////////////////////////////
/////////////////////////////////////////////
/////////////////////////////////////////////
/////////////////////////////////////////////
// Below here are the implementation guts  //
/////////////////////////////////////////////
/////////////////////////////////////////////
/////////////////////////////////////////////
/////////////////////////////////////////////

bool do_fft_3d(int N1,int N2,int N3,double *out,double *in,int num_threads) {
	if (num_threads>1) {
		fftw_init_threads();
		fftw_plan_with_nthreads(num_threads);
	}

	int N1N2N3=N1*N2*N3;
	fftw_complex *in2=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*N1N2N3);
	fftw_complex *out2=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*N1N2N3);
	for (int ii=0; ii<N1N2N3; ii++) {
		in2[ii][0]=in[ii*2];
		in2[ii][1]=in[ii*2+1];
	}
	fftw_plan p=fftw_plan_dft_3d(N3,N2,N1,in2,out2,FFTW_BACKWARD,FFTW_ESTIMATE); //do we need to do N3,N2,N1 backwards? Really? Wasted half of a day. Oh well, in the mean time I made sure the rest of my code was bug free.
	fftw_execute(p);
	for (int ii=0; ii<N1N2N3; ii++) {
		out[ii*2]=out2[ii][0];
		out[ii*2+1]=out2[ii][1];
	}
	fftw_free(in2);
	fftw_free(out2);

	if (num_threads>1) {
		fftw_cleanup_threads();
	}

	return true;
}

void do_fix_3d_OLD(int N1,int N2,int N3,int M,KernelInfo KK1,KernelInfo KK2,KernelInfo KK3,double *out,double *out_oversamp_hat) {

    double oversamp=KK1.oversamp;

    int N1b=(int)(N1*oversamp);
    int N2b=(int)(N2*oversamp);
    int N3b=(int)(N3*oversamp);

#ifdef NUFFT_ANALYTIC_CORRECTION

    double lambda=M_PI/KK.tau;

    double *correction_vals1=(double *)malloc(sizeof(double)*(N1/2+2));
    double *correction_vals2=(double *)malloc(sizeof(double)*(N2/2+2));
    double *correction_vals3=(double *)malloc(sizeof(double)*(N3/2+2));

    double t1=M_PI * lambda / (N1b*N1b);
    double t2=M_PI * lambda / (N2b*N2b);
    double t3=M_PI * lambda / (N3b*N3b);
    for (int i=0; i<N1/2+2; i++) {
        correction_vals1[i]=exp(-i*i*t1)*(lambda*sqrt(lambda))*M;
    }
    for (int i=0; i<N2/2+2; i++) {
        correction_vals2[i]=exp(-i*i*t2);
    }
    for (int i=0; i<N3/2+2; i++) {
        correction_vals3[i]=exp(-i*i*t3);
    }

    for (int i3=0; i3<N3; i3++) {
        int aa3=((i3+N3/2)%N3)*N1*N2; //this includes a shift of the zero frequency
        int bb3=0;
        double correction3=1/(lambda*sqrt(lambda));
        if (i3<(N3+1)/2) { //had to be careful here!
            bb3=i3*N1*oversamp*N2*oversamp;
            correction3=correction_vals3[i3];
        }
        else {
            bb3=(N3*oversamp-(N3-i3))*N1*oversamp*N2*oversamp;
            correction3=correction_vals3[N3-i3];
        }
        for (int i2=0; i2<N2; i2++) {
            int aa2=((i2+N2/2)%N2)*N1;
            int bb2=0;
            double correction2=1;
            if (i2<(N2+1)/2) {
                bb2=i2*N1*oversamp;
                correction2=correction_vals2[i2];
            }
            else {
                bb2=(N2*oversamp-(N2-i2))*N1*oversamp;
                correction2=correction_vals2[N2-i2];
            }
            aa2+=aa3;
            bb2+=bb3;
            correction2*=correction3;
            for (int i1=0; i1<N1; i1++) {
                int aa1=(i1+N1/2)%N1;
                int bb1=0;
                double correction1=1;
                if (i1<(N1+1)/2) {
                    bb1=i1;
                    correction1=correction_vals1[i1];
                }
                else {
                    bb1=N1*oversamp-(N1-i1);
                    correction1=correction_vals1[N1-i1];
                }
                aa1+=aa2;
                bb1+=bb2;
                correction1*=correction2;
                out[aa1*2]=out_oversamp_hat[bb1*2]/correction1;
                out[aa1*2+1]=out_oversamp_hat[bb1*2+1]/correction1;
            }
        }
    }
    free(correction_vals1);
    free(correction_vals2);
    free(correction_vals3);
#else

    double *xcorrection=(double *)malloc(sizeof(double)*(N1b*2));
    double *ycorrection=(double *)malloc(sizeof(double)*(N2b*2));
    double *zcorrection=(double *)malloc(sizeof(double)*(N3b*2));

    int correction_kernel_oversamp=2; //do not change this -- hard-coded below

    for (int ik=1; ik<=3; ik++) {
        KernelInfo *KK;
        double *correction;
        int Nb;
        if (ik==1) {KK=&KK1; Nb=N1b; correction=xcorrection;}
        if (ik==2) {KK=&KK2; Nb=N2b; correction=ycorrection;}
        if (ik==3) {KK=&KK3; Nb=N3b; correction=zcorrection;}
        int nspread_correction=KK->nspread*2;

        double kernel_A[nspread_correction]; double kernel_B[nspread_correction];
        evaluate_kernel_1d(kernel_A,0,-nspread_correction/2,-nspread_correction/2+nspread_correction-1,*KK);
        evaluate_kernel_1d(kernel_B,-0.5,-nspread_correction/2,-nspread_correction/2+nspread_correction-1,*KK);
        for (int ii=0; ii<Nb*2; ii++) correction[ii]=0;
        for (int dx=-nspread_correction/2; dx<-nspread_correction/2+nspread_correction; dx++) {
            {
                double phase_increment=dx*2*M_PI/Nb;
                double kernel_val=kernel_A[dx+nspread_correction/2]/correction_kernel_oversamp;
                double phase_factor=0;
                for (int xx=0; xx<=Nb/2; xx++) {
                    correction[xx*2]+=kernel_val*cos(phase_factor);
                    correction[xx*2+1]+=kernel_val*sin(phase_factor);
                    phase_factor+=phase_increment;
                }
                phase_factor=-phase_increment;
                for (int xx=Nb-1; xx>Nb/2; xx--) {
                    correction[xx*2]+=kernel_val*cos(phase_factor);
                    correction[xx*2+1]+=kernel_val*sin(phase_factor);
                    phase_factor-=phase_increment;
                }
            }
            {
                double phase_increment=(dx+0.5)*2*M_PI/Nb;
                double kernel_val=kernel_B[dx+nspread_correction/2]/correction_kernel_oversamp;
                double phase_factor=0;
                for (int xx=0; xx<=Nb/2; xx++) {
                    correction[xx*2]+=kernel_val*cos(phase_factor);
                    correction[xx*2+1]+=kernel_val*sin(phase_factor);
                    phase_factor+=phase_increment;
                }
                phase_factor=-phase_increment;
                for (int xx=Nb-1; xx>Nb/2; xx--) {
                    correction[xx*2]+=kernel_val*cos(phase_factor);
                    correction[xx*2+1]+=kernel_val*sin(phase_factor);
                    phase_factor-=phase_increment;
                }
            }
        }
    }

    for (int i3=0; i3<N3; i3++) {
        int aa3=((i3+N3/2)%N3); //this includes a shift of the zero frequency
        int bb3=0;
        if (i3<(N3+1)/2) { //had to be careful here!
            bb3=i3;
        }
        else {
            bb3=(N3*oversamp-(N3-i3));
        }
        double correction3_re=zcorrection[bb3*2];
        double correction3_im=zcorrection[bb3*2+1];
        aa3=aa3*N1*N2;
        bb3=bb3*N1*oversamp*N2*oversamp;
        for (int i2=0; i2<N2; i2++) {
            int aa2=((i2+N2/2)%N2);
            int bb2=0;
            if (i2<(N2+1)/2) {
                bb2=i2;
            }
            else {
                bb2=(N2*oversamp-(N2-i2));
            }
            double correction2_re=correction3_re*ycorrection[bb2*2] - correction3_im*ycorrection[bb2*2+1];
            double correction2_im=correction3_re*ycorrection[bb2*2+1] + correction3_im*ycorrection[bb2*2];
            aa2=aa2*N1+aa3;
            bb2=bb2*N1*oversamp+bb3;
            for (int i1=0; i1<N1; i1++) {
                int aa1=(i1+N1/2)%N1;
                int bb1=0;
                if (i1<(N1+1)/2) {
                    bb1=i1;
                }
                else {
                    bb1=N1*oversamp-(N1-i1);
                }
                double correction1_re=correction2_re*xcorrection[bb1*2] - correction2_im*xcorrection[bb1*2+1];
                double correction1_im=correction2_re*xcorrection[bb1*2+1] + correction2_im*xcorrection[bb1*2];
                double correction1_magsqr=correction1_re*correction1_re+correction1_im*correction1_im;
                double correction1_inv_re=correction1_re/correction1_magsqr;
                double correction1_inv_im=-correction1_im/correction1_magsqr;
                aa1=aa1+aa2;
                bb1=bb1+bb2;
                out[aa1*2]=( out_oversamp_hat[bb1*2]*correction1_inv_re - out_oversamp_hat[bb1*2+1]*correction1_inv_im ) / M;
                out[aa1*2+1]=( out_oversamp_hat[bb1*2]*correction1_inv_im + out_oversamp_hat[bb1*2+1]*correction1_inv_re ) / M;
                //out[aa1*2]=out_oversamp_hat[bb1*2]; //for testing without correction
                //out[aa1*2+1]=out_oversamp_hat[bb1*2+1]; //for testing without correction
            }
        }
    }

    free(xcorrection);
    free(ycorrection);
    free(zcorrection);

#endif
}


void do_fix_3d(int N1,int N2,int N3,int M,KernelInfo KK1,KernelInfo KK2,KernelInfo KK3,double *out,double *out_oversamp_hat) {
    int N1b=(int)(N1*KK1.oversamp);
    int N2b=(int)(N2*KK2.oversamp);
    int N3b=(int)(N3*KK3.oversamp);

#ifdef NUFFT_ANALYTIC_CORRECTION

	double lambda=M_PI/KK.tau;

	double *correction_vals1=(double *)malloc(sizeof(double)*(N1/2+2));
	double *correction_vals2=(double *)malloc(sizeof(double)*(N2/2+2));
	double *correction_vals3=(double *)malloc(sizeof(double)*(N3/2+2));

	double t1=M_PI * lambda / (N1b*N1b);
	double t2=M_PI * lambda / (N2b*N2b);
	double t3=M_PI * lambda / (N3b*N3b);
	for (int i=0; i<N1/2+2; i++) {
		correction_vals1[i]=exp(-i*i*t1)*(lambda*sqrt(lambda))*M;
	}
	for (int i=0; i<N2/2+2; i++) {
		correction_vals2[i]=exp(-i*i*t2);
	}
	for (int i=0; i<N3/2+2; i++) {
		correction_vals3[i]=exp(-i*i*t3);
	}

	for (int i3=0; i3<N3; i3++) {
		int aa3=((i3+N3/2)%N3)*N1*N2; //this includes a shift of the zero frequency
		int bb3=0;
		double correction3=1/(lambda*sqrt(lambda));
		if (i3<(N3+1)/2) { //had to be careful here!
            bb3=i3*N1*KK1.oversamp*N2*KK2.oversamp;
			correction3=correction_vals3[i3];
		}
		else {
            bb3=(N3*KK3.oversamp-(N3-i3))*N1*KK1.oversamp*N2*KK2.oversamp;
			correction3=correction_vals3[N3-i3];
		}
		for (int i2=0; i2<N2; i2++) {
			int aa2=((i2+N2/2)%N2)*N1;
			int bb2=0;
			double correction2=1;
			if (i2<(N2+1)/2) {
                bb2=i2*N1*KK1.oversamp;
				correction2=correction_vals2[i2];
			}
			else {
                bb2=(N2*KK2.oversamp-(N2-i2))*N1*KK1.oversamp;
				correction2=correction_vals2[N2-i2];
			}
			aa2+=aa3;
			bb2+=bb3;
			correction2*=correction3;
			for (int i1=0; i1<N1; i1++) {
				int aa1=(i1+N1/2)%N1;
				int bb1=0;
				double correction1=1;
				if (i1<(N1+1)/2) {
					bb1=i1;
					correction1=correction_vals1[i1];
				}
				else {
                    bb1=N1*KK1.oversamp-(N1-i1);
					correction1=correction_vals1[N1-i1];
				}
				aa1+=aa2;
				bb1+=bb2;
				correction1*=correction2;
				out[aa1*2]=out_oversamp_hat[bb1*2]/correction1;
				out[aa1*2+1]=out_oversamp_hat[bb1*2+1]/correction1;
			}
		}
	}
	free(correction_vals1);
	free(correction_vals2);
	free(correction_vals3);
#else

    #ifdef USE_COMPLEX_CORRECTION
	double *xcorrection=(double *)malloc(sizeof(double)*(N1b*2));
	double *ycorrection=(double *)malloc(sizeof(double)*(N2b*2));
	double *zcorrection=(double *)malloc(sizeof(double)*(N3b*2));
    #else
    double *xcorrection=(double *)malloc(sizeof(double)*(N1b));
    double *ycorrection=(double *)malloc(sizeof(double)*(N2b));
    double *zcorrection=(double *)malloc(sizeof(double)*(N3b));
    #endif

	int correction_kernel_oversamp=2; //do not change this -- hard-coded below

	for (int ik=1; ik<=3; ik++) {
		KernelInfo *KK;
		double *correction;
		int Nb;
		if (ik==1) {KK=&KK1; Nb=N1b; correction=xcorrection;}
		if (ik==2) {KK=&KK2; Nb=N2b; correction=ycorrection;}
		if (ik==3) {KK=&KK3; Nb=N3b; correction=zcorrection;}
		int nspread_correction=KK->nspread*2;

		double kernel_A[nspread_correction]; double kernel_B[nspread_correction];
		evaluate_kernel_1d(kernel_A,0,-nspread_correction/2,-nspread_correction/2+nspread_correction-1,*KK);
		evaluate_kernel_1d(kernel_B,-0.5,-nspread_correction/2,-nspread_correction/2+nspread_correction-1,*KK);
        #ifdef USE_COMPLEX_CORRECTION
        for (int ii=0; ii<Nb*2; ii++) correction[ii]=0;
        #else
        for (int ii=0; ii<Nb; ii++) correction[ii]=0;
        #endif
		for (int dx=-nspread_correction/2; dx<-nspread_correction/2+nspread_correction; dx++) {
			{
				double phase_increment=dx*2*M_PI/Nb;
				double kernel_val=kernel_A[dx+nspread_correction/2]/correction_kernel_oversamp;
				double phase_factor=0;
				for (int xx=0; xx<=Nb/2; xx++) {
                    #ifdef USE_COMPLEX_CORRECTION
					correction[xx*2]+=kernel_val*cos(phase_factor);
					correction[xx*2+1]+=kernel_val*sin(phase_factor);
                    #else
                    correction[xx]+=kernel_val*cos(phase_factor);
                    #endif
                    phase_factor+=phase_increment;
				}
				phase_factor=-phase_increment;
				for (int xx=Nb-1; xx>Nb/2; xx--) {
                    #ifdef USE_COMPLEX_CORRECTION
					correction[xx*2]+=kernel_val*cos(phase_factor);
					correction[xx*2+1]+=kernel_val*sin(phase_factor);
                    #else
                    correction[xx]+=kernel_val*cos(phase_factor);
                    #endif
                    phase_factor-=phase_increment;
				}
			}
			{
				double phase_increment=(dx+0.5)*2*M_PI/Nb;
				double kernel_val=kernel_B[dx+nspread_correction/2]/correction_kernel_oversamp;
				double phase_factor=0;
				for (int xx=0; xx<=Nb/2; xx++) {
                    #ifdef USE_COMPLEX_CORRECTION
					correction[xx*2]+=kernel_val*cos(phase_factor);
					correction[xx*2+1]+=kernel_val*sin(phase_factor);
                    #else
                    correction[xx]+=kernel_val*cos(phase_factor);
                    #endif
                    phase_factor+=phase_increment;
				}
				phase_factor=-phase_increment;
				for (int xx=Nb-1; xx>Nb/2; xx--) {
                    #ifdef USE_COMPLEX_CORRECTION
					correction[xx*2]+=kernel_val*cos(phase_factor);
					correction[xx*2+1]+=kernel_val*sin(phase_factor);
                    #else
                    correction[xx]+=kernel_val*cos(phase_factor);
                    #endif
					phase_factor-=phase_increment;
				}
			}
        }
        for (int ii=0; ii<Nb; ii++) {
            #ifdef USE_COMPLEX_CORRECTION
            double magsqr=correction[2*ii]*correction[2*ii]+correction[2*ii+1]*correction[2*ii+1];
            double inv_re=correction[2*ii]/magsqr;
            double inv_im=-correction[2*ii+1]/magsqr;
            correction[2*ii]=inv_re;
            correction[2*ii+1]=inv_im;
            #else
            correction[ii]=1/correction[ii];
            #endif
        }
    }

	for (int i3=0; i3<N3; i3++) {
		int aa3=((i3+N3/2)%N3); //this includes a shift of the zero frequency
		int bb3=0;
		if (i3<(N3+1)/2) { //had to be careful here!
			bb3=i3;
		}
		else {
            bb3=(N3*KK3.oversamp-(N3-i3));
		}
        #ifdef USE_COMPLEX_CORRECTION
		double correction3_re=zcorrection[bb3*2];
		double correction3_im=zcorrection[bb3*2+1];
        #else
        double correction3=zcorrection[bb3];
        #endif
		aa3=aa3*N1*N2;
        bb3=bb3*N1*KK1.oversamp*N2*KK2.oversamp;
		for (int i2=0; i2<N2; i2++) {
			int aa2=((i2+N2/2)%N2);
			int bb2=0;
			if (i2<(N2+1)/2) {
				bb2=i2;
			}
			else {
                bb2=(N2*KK2.oversamp-(N2-i2));
			}
            #ifdef USE_COMPLEX_CORRECTION
			double correction2_re=correction3_re*ycorrection[bb2*2] - correction3_im*ycorrection[bb2*2+1];
			double correction2_im=correction3_re*ycorrection[bb2*2+1] + correction3_im*ycorrection[bb2*2];
            #else
            double correction2=correction3*ycorrection[bb2];
            #endif
			aa2=aa2*N1+aa3;
            bb2=bb2*N1*KK1.oversamp+bb3;
			for (int i1=0; i1<N1; i1++) {
				int aa1=(i1+N1/2)%N1;
				int bb1=0;
				if (i1<(N1+1)/2) {
					bb1=i1;
				}
				else {
                    bb1=N1*KK1.oversamp-(N1-i1);
				}
                #ifdef USE_COMPLEX_CORRECTION
				double correction1_re=correction2_re*xcorrection[bb1*2] - correction2_im*xcorrection[bb1*2+1];
				double correction1_im=correction2_re*xcorrection[bb1*2+1] + correction2_im*xcorrection[bb1*2];
                #else
                double correction1=correction2*xcorrection[bb1];
                #endif

				aa1=aa1+aa2;
				bb1=bb1+bb2;
                #ifdef USE_COMPLEX_CORRECTION
                out[aa1*2]=( out_oversamp_hat[bb1*2]*correction1_re - out_oversamp_hat[bb1*2+1]*correction1_im ) / M;
                out[aa1*2+1]=( out_oversamp_hat[bb1*2]*correction1_im + out_oversamp_hat[bb1*2+1]*correction1_re ) / M;
                #else
                out[aa1*2]=out_oversamp_hat[bb1*2]*correction1/M;
                out[aa1*2+1]=out_oversamp_hat[bb1*2+1]*correction1/M;
                #endif
			}
		}
	}

	free(xcorrection);
	free(ycorrection);
	free(zcorrection);

#endif
}

