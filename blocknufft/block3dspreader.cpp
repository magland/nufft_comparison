#include "block3dspreader.h"
#include "qute.h"
#include <stdio.h>
#include <math.h>
#include "besseli.h"
#include "omp.h"
#include "unistd.h"
#include <stdlib.h>
#include <sys/wait.h>

// The following is needed because incredibly C++ does not seem to have a round() function
#define ROUND_2_INT(f) ((int)(f >= 0.0 ? (f + 0.5) : (f - 0.5)))

//these are redundantly defined for now! (see blocknufft3d.h)
#define KERNEL_TYPE_GAUSSIAN		1
#define KERNEL_TYPE_KB				2

struct PrecomputeData {
    int *xmins,*ymins,*zmins; //nearest grid points
    int *xmaxs,*ymaxs,*zmaxs; //nearest grid points
    int *iixs,*iiys,*iizs; //nearest grid points
    double *xkernel,*ykernel,*zkernel;
};

PrecomputeData *newEmptyPrecomputeData() {
    PrecomputeData *ret=new PrecomputeData;
    ret->xmins=ret->ymins=ret->zmins=0;
    ret->xmaxs=ret->ymaxs=ret->zmaxs=0;
    ret->iixs=ret->iiys=ret->iizs=0;
    ret->xkernel=ret->ykernel=ret->zkernel=0;
    return ret;
}

void free_precompute_data(PrecomputeData *PD) {
    if (PD->xmins) free(PD->xmins); PD->xmins=0;
    if (PD->ymins) free(PD->ymins); PD->ymins=0;
    if (PD->zmins) free(PD->zmins); PD->zmins=0;
    if (PD->xmaxs) free(PD->xmaxs); PD->xmaxs=0;
    if (PD->ymaxs) free(PD->ymaxs); PD->ymaxs=0;
    if (PD->zmaxs) free(PD->zmaxs); PD->zmaxs=0;
    if (PD->iixs) free(PD->iixs); PD->iixs=0;
    if (PD->iiys) free(PD->iiys); PD->iiys=0;
    if (PD->iizs) free(PD->iizs); PD->iizs=0;
    if (PD->xkernel) free(PD->xkernel); PD->xkernel=0;
    if (PD->ykernel) free(PD->ykernel); PD->ykernel=0;
    if (PD->zkernel) free(PD->zkernel); PD->zkernel=0;
}

void free_block_data(BlockData *BD) {
    free(BD->x);
    free(BD->y);
    free(BD->z);
    free(BD->nonuniform_d);
    free(BD->uniform_d);
    free(BD->nonuniform_indices);
}

class Block3DSpreaderPrivate {
public:
	Block3DSpreader *q;
	QList<BlockData *> m_blocks;
    QList<PrecomputeData *> m_precompute_data; //one for every block
	int m_parallel_type;
	int m_num_threads;
    int m_N1,m_N2,m_N3;
    int m_M;
    int m_kernel_type;
    KernelInfo KK1,KK2,KK3;

	void blockspread3d(BlockData *BD);
};
void precompute_block(BlockData *BD,PrecomputeData *PD,KernelInfo KK1,KernelInfo KK2,KernelInfo KK3);

Block3DSpreader::Block3DSpreader()
{
	d=new Block3DSpreaderPrivate;
	d->q=this;
	d->m_parallel_type=PARALLEL_NONE;
	d->m_num_threads=1;
    d->m_kernel_type=0;
    d->m_N1=d->m_N2=d->m_N3=0;
    d->m_M=0;
}

Block3DSpreader::~Block3DSpreader()
{
    for (int i=0; i<d->m_precompute_data.count(); i++) {
        free_precompute_data(d->m_precompute_data.value(i));
        delete d->m_precompute_data.value(i);
    }
    for (int i=0; i<d->m_precompute_data.count(); i++) {
        free_block_data(d->m_blocks.value(i));
        delete d->m_blocks.value(i);
    }
    delete d;
}

void Block3DSpreader::setKernelInfo(KernelInfo KK1,KernelInfo KK2,KernelInfo KK3)
{
    d->KK1=KK1;
    d->KK2=KK2;
    d->KK3=KK3;
}

void Block3DSpreader::setNumThreads(int num)
{
    d->m_num_threads=num;
}

void Block3DSpreader::setN(int N1, int N2, int N3)
{
    d->m_N1=N1;
    d->m_N2=N2;
    d->m_N3=N3;
}

void Block3DSpreader::setM(int M)
{
    d->m_M=M;
}

bool check_valid_inputs(BlockData *BD) {
	for (int m=0; m<BD->M; m++) {
		if ((BD->x[m]<0)||(BD->x[m]>=BD->N1o)) {
            printf ("Out of range: %g, %d\n",BD->x[m],BD->N1o);
			return false;
		}
		if ((BD->y[m]<0)||(BD->y[m]>=BD->N2o)) {
            printf ("Out of range: %g, %d\n",BD->y[m],BD->N2o);
			return false;
		}
		if ((BD->z[m]<0)||(BD->z[m]>=BD->N3o)) {
            printf ("Out of range: %g, %d\n",BD->z[m],BD->N3o);
			return false;
		}
	}
	return true;
}

void evaluate_kernel_1d(double *out,double diff,int imin,int imax,const KernelInfo &KK) {

	if (KK.kernel_type==KERNEL_TYPE_GAUSSIAN) {
		//exp(-(dx-i)^2*tau) = exp(-dx^2*tau)*exp(2*dx*tau)^i*exp(-i*i*tau)
		double term1=exp(-diff*diff*KK.tau);
		double factor2=exp(2*diff*KK.tau);
		double factor2_inv=1/factor2;

		double term2=term1;
		for (int i=0; i<=imax; i++) {
			out[i-imin]=term2*KK.lookup_exp[i];
			term2*=factor2;
		}

		term2=term1*factor2_inv;
		for (int i=-1; i>=imin; i--) {
			out[i-imin]=term2*KK.lookup_exp[-i];
			term2*=factor2_inv;
		}
	}
	else if (KK.kernel_type==KERNEL_TYPE_KB) {
		for (int ii=imin; ii<=imax; ii++) {
            //out[ii-imin]=1;

			double x=diff-ii;
			double tmp1=1-(2*x/KK.W)*(2*x/KK.W);
			if (tmp1<0) {
				out[ii-imin]=0;
			}
			else {
				double y=KK.beta*sqrt(tmp1);
                //out[ii-imin]=besseli0(y);
                out[ii-imin]=besseli0_approx(y);
			}

		}
	}
}

void do_spreading(BlockData *BD,PrecomputeData *PD,double *uniform_d,double *nonuniform_d,KernelInfo KK1,KernelInfo KK2,KernelInfo KK3) {
	double N1o_times_2=BD->N1o*2;
	double N1oN2o_times_2=BD->N1o*BD->N2o*2;
	double N1oN2oN3o_times_2=BD->N1o*BD->N2o*BD->N3o*2;
	for (int ii=0; ii<N1oN2oN3o_times_2; ii++) BD->uniform_d[ii]=0;
	for (int jj=0; jj<BD->M; jj++) {
        double d0_re=nonuniform_d[jj*2],d0_im=nonuniform_d[jj*2+1];

        int xmin=PD->xmins[jj];
        int xmax=PD->xmaxs[jj];
        int iix=PD->iixs[jj];
        int ymin=PD->ymins[jj];
        int ymax=PD->ymaxs[jj];
        int iiy=PD->iiys[jj];
        int zmin=PD->zmins[jj];
        int zmax=PD->zmaxs[jj];
        int iiz=PD->iizs[jj];

        double *x_kernel=&PD->xkernel[jj*(KK1.nspread+1)];
        double *y_kernel=&PD->ykernel[jj*(KK2.nspread+1)];
        double *z_kernel=&PD->zkernel[jj*(KK3.nspread+1)];

		int xmax_minus_xmin_plus_iix=xmax-xmin+iix;
		int xmin_times_2=xmin*2;
		for (int iz=zmin; iz<=zmax; iz++) {
			double kernval_00=z_kernel[iz-zmin+iiz];
			int kkk1=N1oN2o_times_2*iz; //complex index
			int kkk2=kkk1+N1o_times_2*ymin; //remember complex
			for (int iy=ymin; iy<=ymax; iy++) {
				double kernval_01=kernval_00*y_kernel[iy-ymin+iiy];
				double kernval_01_times_d0_re=kernval_01*d0_re;
				double kernval_01_times_d0_im=kernval_01*d0_im;
				int kkk3=kkk2+xmin_times_2; //remember it is complex
				for (int ix0=iix; ix0<=xmax_minus_xmin_plus_iix; ix0++) {
					//most of the time is spent inside this inner-most loop
                    uniform_d[kkk3]+=x_kernel[ix0]*kernval_01_times_d0_re;
                    uniform_d[kkk3+1]+=x_kernel[ix0]*kernval_01_times_d0_im;
					kkk3+=2; //remember it is complex
				}
				kkk2+=N1o_times_2; //remember complex
			}
		}

	}
}

// Here's the spreading!
bool blockspread3d(BlockData *BD,PrecomputeData *PD,double *uniform_d,double *nonuniform_d,KernelInfo KK1,KernelInfo KK2,KernelInfo KK3) {
    QTime timer0; timer0.start();

    bool precomputed=false;
    if (PD->xkernel) precomputed=true;
    if (!precomputed) {
        precompute_block(BD,PD,KK1,KK2,KK3);
    }

	// Transfer opts and other parameters to D

	//Check to see if we have valid inputs
	if (!check_valid_inputs(BD)) {
		for (int i=0; i<BD->N1o*BD->N2o*BD->N3o*2; i++) BD->uniform_d[i]=0;
		return false;
	}

    do_spreading(BD,PD,uniform_d,nonuniform_d,KK1,KK2,KK3);
    //printf ("############# elapsed for blockspread3d (%d x %d x %d) M=%d: %d\n",BD->N1o,BD->N2o,BD->N3o,BD->M,timer0.elapsed());

    if (!precomputed) {
        free_precompute_data(PD);
    }

	return true;
}


void Block3DSpreader::addBlock(BlockData *B)
{
	d->m_blocks.append(B);
    d->m_precompute_data.append(newEmptyPrecomputeData());
}

void Block3DSpreader::setParallel(int parallel_type, int num_threads)
{
	d->m_parallel_type=parallel_type;
    d->m_num_threads=num_threads;
}

void Block3DSpreader::precompute()
{
    int num_blocks=d->m_blocks.count();
    if (d->m_parallel_type==PARALLEL_NONE) {
        for (int i=0; i<num_blocks; i++) {
            BlockData *BD=d->m_blocks.value(i);
            PrecomputeData *PD=d->m_precompute_data.value(i);
            precompute_block(BD,PD,d->KK1,d->KK2,d->KK3);
        }
    }
    else if (d->m_parallel_type==PARALLEL_OPENMP) {
        omp_set_num_threads(d->m_num_threads);
        #pragma omp parallel
        {
            #pragma omp for
            for (int i=0; i<num_blocks; i++) {
                BlockData *BD=d->m_blocks.value(i);
                PrecomputeData *PD=d->m_precompute_data.value(i);
                precompute_block(BD,PD,d->KK1,d->KK2,d->KK3);
            }
        }
    }


}

void precompute_block(BlockData *BD, PrecomputeData *PD,KernelInfo KK1,KernelInfo KK2,KernelInfo KK3)
{
    free_precompute_data(PD);

    PD->xkernel=(double *)malloc(sizeof(double)*(KK1.nspread+1)*BD->M);
    PD->ykernel=(double *)malloc(sizeof(double)*(KK2.nspread+1)*BD->M);
    PD->zkernel=(double *)malloc(sizeof(double)*(KK3.nspread+1)*BD->M);
    PD->xmins=(int *)malloc(sizeof(double)*BD->M);
    PD->ymins=(int *)malloc(sizeof(double)*BD->M);
    PD->zmins=(int *)malloc(sizeof(double)*BD->M);
    PD->xmaxs=(int *)malloc(sizeof(double)*BD->M);
    PD->ymaxs=(int *)malloc(sizeof(double)*BD->M);
    PD->zmaxs=(int *)malloc(sizeof(double)*BD->M);
    PD->iixs=(int *)malloc(sizeof(double)*BD->M);
    PD->iiys=(int *)malloc(sizeof(double)*BD->M);
    PD->iizs=(int *)malloc(sizeof(double)*BD->M);

    for (int i=0; i<BD->M; i++) {
        double x0=BD->x[i];
        double y0=BD->y[i];
        double z0=BD->z[i];

        int x_integer=ROUND_2_INT(x0);
        double x_diff=x0-x_integer;
        int y_integer=ROUND_2_INT(y0);
        double y_diff=y0-y_integer;
        int z_integer=ROUND_2_INT(z0);
        double z_diff=z0-z_integer;

        int xmin,xmax;
        if (x_diff<0) {
            xmin=fmax(x_integer-KK1.nspread/2,0);
            xmax=fmin(x_integer+KK1.nspread/2-1,BD->N1o-1);
        }
        else {
            xmin=fmax(x_integer-KK1.nspread/2+1,0);
            xmax=fmin(x_integer+KK1.nspread/2,BD->N1o-1);
        }
        int iix=xmin-(x_integer-KK1.nspread/2);

        int ymin,ymax;
        if (y_diff<0) {
            ymin=fmax(y_integer-KK2.nspread/2,0);
            ymax=fmin(y_integer+KK2.nspread/2-1,BD->N2o-1);
        }
        else {
            ymin=fmax(y_integer-KK2.nspread/2+1,0);
            ymax=fmin(y_integer+KK2.nspread/2,BD->N2o-1);
        }
        int iiy=ymin-(y_integer-KK2.nspread/2);

        int zmin,zmax;
        if (z_diff<0) {
            zmin=fmax(z_integer-KK3.nspread/2,0);
            zmax=fmin(z_integer+KK3.nspread/2-1,BD->N3o-1);
        }
        else {
            zmin=fmax(z_integer-KK3.nspread/2+1,0);
            zmax=fmin(z_integer+KK3.nspread/2,BD->N3o-1);
        }
        int iiz=zmin-(z_integer-KK3.nspread/2);

        double *x_kernel=&PD->xkernel[(KK1.nspread+1)*i];
        double *y_kernel=&PD->ykernel[(KK2.nspread+1)*i];
        double *z_kernel=&PD->zkernel[(KK3.nspread+1)*i];

        evaluate_kernel_1d(x_kernel,x_diff,-KK1.nspread/2,-KK1.nspread/2+KK1.nspread,KK1);
        evaluate_kernel_1d(y_kernel,y_diff,-KK2.nspread/2,-KK2.nspread/2+KK2.nspread,KK2);
        evaluate_kernel_1d(z_kernel,z_diff,-KK3.nspread/2,-KK3.nspread/2+KK3.nspread,KK3);

        PD->xmins[i]=xmin;
        PD->xmaxs[i]=xmax;
        PD->iixs[i]=iix;
        PD->ymins[i]=ymin;
        PD->ymaxs[i]=ymax;
        PD->iiys[i]=iiy;
        PD->zmins[i]=zmin;
        PD->zmaxs[i]=zmax;
        PD->iizs[i]=iiz;
    }
}

void Block3DSpreader::run()
{
	int num_blocks=d->m_blocks.count();
	if (d->m_parallel_type==PARALLEL_NONE) {
		for (int bb=0; bb<num_blocks; bb++) {
			BlockData *BD=d->m_blocks.value(bb);
            PrecomputeData *PD=d->m_precompute_data.value(bb);
            blockspread3d(BD,PD,BD->uniform_d,BD->nonuniform_d,d->KK1,d->KK2,d->KK3);
		}
	}
	else if (d->m_parallel_type==PARALLEL_OPENMP) {
		omp_set_num_threads(d->m_num_threads);
		#pragma omp parallel
		{
			#pragma omp for
			for (int bb=0; bb<num_blocks; bb++) {
				BlockData *BD=d->m_blocks.value(bb);
                PrecomputeData *PD=d->m_precompute_data.value(bb);
                blockspread3d(BD,PD,BD->uniform_d,BD->nonuniform_d,d->KK1,d->KK2,d->KK3);
			}
		}
    }
}

int Block3DSpreader::blockCount()
{
    return d->m_blocks.count();
}

BlockData *Block3DSpreader::block(int ind)
{
    return d->m_blocks.value(ind);
}

KernelInfo Block3DSpreader::KK1()
{
    return d->KK1;
}
KernelInfo Block3DSpreader::KK2()
{
    return d->KK2;
}
KernelInfo Block3DSpreader::KK3()
{
    return d->KK3;
}

int Block3DSpreader::numThreads()
{
    return d->m_num_threads;
}

int Block3DSpreader::N1()
{
    return d->m_N1;
}
int Block3DSpreader::N2()
{
    return d->m_N2;
}
int Block3DSpreader::N3()
{
    return d->m_N3;
}

int Block3DSpreader::M()
{
    return d->m_M;
}
