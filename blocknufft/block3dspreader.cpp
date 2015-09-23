#include "block3dspreader.h"
#include "qute.h"
#include <stdio.h>
#include <math.h>
#include "besseli.h"
#include "omp.h"

// The following is needed because incredibly C++ does not seem to have a round() function
#define ROUND_2_INT(f) ((int)(f >= 0.0 ? (f + 0.5) : (f - 0.5)))

//these are redundantly defined for now! (see blocknufft3d.h)
#define KERNEL_TYPE_GAUSSIAN		1
#define KERNEL_TYPE_KB				2

class Block3DSpreaderPrivate {
public:
	Block3DSpreader *q;
	QList<BlockData *> m_blocks;
	int m_parallel_type;
	int m_num_threads;

	void blockspread3d(BlockData *BD);

};

Block3DSpreader::Block3DSpreader()
{
	d=new Block3DSpreaderPrivate;
	d->q=this;
	d->m_parallel_type=PARALLEL_NONE;
	d->m_num_threads=1;
}

Block3DSpreader::~Block3DSpreader()
{
	delete d;
}

bool check_valid_inputs(BlockData *BD) {
	for (int m=0; m<BD->M; m++) {
		if ((BD->x[m]<0)||(BD->x[m]>=BD->N1o)) {
			printf("Out of range: %g, %d\n",BD->x[m],BD->N1o);
			return false;
		}
		if ((BD->y[m]<0)||(BD->y[m]>=BD->N2o)) {
			printf("Out of range: %g, %d\n",BD->y[m],BD->N2o);
			return false;
		}
		if ((BD->z[m]<0)||(BD->z[m]>=BD->N3o)) {
			printf("Out of range: %g, %d\n",BD->z[m],BD->N3o);
			return false;
		}
	}

	printf("inputs are okay.\n");
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
				out[ii-imin]=besseli0_approx(y);
			}
		}
	}
}


void do_spreading(BlockData *BD) {
	double x_kernel[BD->KK1->nspread+1];
	double y_kernel[BD->KK2->nspread+1];
	double z_kernel[BD->KK3->nspread+1];

	double N1o_times_2=BD->N1o*2;
	double N1oN2o_times_2=BD->N1o*BD->N2o*2;
	double N1oN2oN3o_times_2=BD->N1o*BD->N2o*BD->N3o*2;
	for (int ii=0; ii<N1oN2oN3o_times_2; ii++) BD->uniform_d[ii]=0;
	for (int jj=0; jj<BD->M; jj++) {
		double x0=BD->x[jj],y0=BD->y[jj],z0=BD->z[jj],d0_re=BD->nonuniform_d[jj*2],d0_im=BD->nonuniform_d[jj*2+1];

		int x_integer=ROUND_2_INT(x0);
		double x_diff=x0-x_integer;
		int y_integer=ROUND_2_INT(y0);
		double y_diff=y0-y_integer;
		int z_integer=ROUND_2_INT(z0);
		double z_diff=z0-z_integer;

		evaluate_kernel_1d(x_kernel,x_diff,-BD->KK1->nspread/2,-BD->KK1->nspread/2+BD->KK1->nspread,*BD->KK1);
		evaluate_kernel_1d(y_kernel,y_diff,-BD->KK2->nspread/2,-BD->KK2->nspread/2+BD->KK2->nspread,*BD->KK2);
		evaluate_kernel_1d(z_kernel,z_diff,-BD->KK3->nspread/2,-BD->KK3->nspread/2+BD->KK3->nspread,*BD->KK3);

		int xmin,xmax;
		if (x_diff<0) {
			xmin=fmax(x_integer-BD->KK1->nspread/2,0);
			xmax=fmin(x_integer+BD->KK1->nspread/2-1,BD->N1o-1);
		}
		else {
			xmin=fmax(x_integer-BD->KK1->nspread/2+1,0);
			xmax=fmin(x_integer+BD->KK1->nspread/2,BD->N1o-1);
		}
		int iix=xmin-(x_integer-BD->KK1->nspread/2);


		int ymin,ymax;
		if (y_diff<0) {
			ymin=fmax(y_integer-BD->KK2->nspread/2,0);
			ymax=fmin(y_integer+BD->KK2->nspread/2-1,BD->N2o-1);
		}
		else {
			ymin=fmax(y_integer-BD->KK2->nspread/2+1,0);
			ymax=fmin(y_integer+BD->KK2->nspread/2,BD->N2o-1);
		}
		int iiy=ymin-(y_integer-BD->KK2->nspread/2);

		int zmin,zmax;
		if (z_diff<0) {
			zmin=fmax(z_integer-BD->KK3->nspread/2,0);
			zmax=fmin(z_integer+BD->KK3->nspread/2-1,BD->N3o-1);
		}
		else {
			zmin=fmax(z_integer-BD->KK3->nspread/2+1,0);
			zmax=fmin(z_integer+BD->KK3->nspread/2,BD->N3o-1);
		}
		int iiz=zmin-(z_integer-BD->KK3->nspread/2);

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
					BD->uniform_d[kkk3]+=x_kernel[ix0]*kernval_01_times_d0_re;
					BD->uniform_d[kkk3+1]+=x_kernel[ix0]*kernval_01_times_d0_im;
					kkk3+=2; //remember it is complex
				}
				kkk2+=N1o_times_2; //remember complex
			}
		}
	}
}



// Here's the spreading!
bool blockspread3d(BlockData *BD) {
	QTime timer0;
	printf("Starting blockspread3d...\n");

	// Transfer opts and other parameters to D

	//Check to see if we have valid inputs
	printf("Checking inputs...\n"); timer0.start();
	if (!check_valid_inputs(BD)) {
		for (int i=0; i<BD->N1o*BD->N2o*BD->N3o*2; i++) BD->uniform_d[i]=0;
		return false;
	}
	printf("  --- Elapsed: %d ms\n",timer0.elapsed());

	printf("spreading...\n"); timer0.start();
	do_spreading(BD);
	printf("  --- Elapsed: %d ms\n",timer0.elapsed());

	return true;
}


void Block3DSpreader::addBlock(BlockData *B)
{
	d->m_blocks.append(B);
}

void Block3DSpreader::setParallel(int parallel_type, int num_threads)
{
	d->m_parallel_type=parallel_type;
	d->m_num_threads=num_threads;
}

void Block3DSpreader::run()
{
	int num_blocks=d->m_blocks.count();
	if (d->m_parallel_type==PARALLEL_NONE) {
		for (int bb=0; bb<num_blocks; bb++) {
			BlockData *BD=d->m_blocks.value(bb);
			blockspread3d(BD);
		}
	}
	else if (d->m_parallel_type==PARALLEL_OPENMP) {
		omp_set_num_threads(d->m_num_threads);
		#pragma omp parallel
		{
			#pragma omp for
			for (int bb=0; bb<num_blocks; bb++) {
				BlockData *BD=d->m_blocks.value(bb);
				blockspread3d(BD);
			}
		}
	}
}


