#ifndef BLOCK3DSPREADER_H
#define BLOCK3DSPREADER_H

// Info for the spreading kernel
struct KernelInfo {
	int kernel_type;

    double oversamp;
	int nspread;

	//for gaussian
	double tau;
	double *lookup_exp;

	//for KB
	double W;
	double beta;
};

struct BlockData {
	int xmin,ymin,zmin;
	int xmax,ymax,zmax;
	int M;
	int N1o,N2o,N3o;
	double *x,*y,*z;
	double *nonuniform_d;
	double *uniform_d;
    int *nonuniform_indices;
	int jj;
    int x_block_index,y_block_index,z_block_index;
};

#define PARALLEL_NONE 0
#define PARALLEL_OPENMP 1
#define PARALLEL_FORK 2

class Block3DSpreaderPrivate;
class Block3DSpreader
{
public:
	friend class Block3DSpreaderPrivate;
	Block3DSpreader();
	virtual ~Block3DSpreader();
    void setKernelInfo(KernelInfo KK1,KernelInfo KK2,KernelInfo KK3);
    void setNumThreads(int num);
    void setN(int N1,int N2,int N3); //for information in the run
    void setM(int M); //for information in the run
	void addBlock(BlockData *B);
	void setParallel(int parallel_type,int num_threads);
    void precompute();
    void run();

    int blockCount();
    BlockData *block(int ind);
    KernelInfo KK1();
    KernelInfo KK2();
    KernelInfo KK3();
    int numThreads();
    int N1();
    int N2();
    int N3();
    int M();
private:
	Block3DSpreaderPrivate *d;
};

void evaluate_kernel_1d(double *out,double diff,int imin,int imax,const KernelInfo &KK);

#endif // BLOCK3DSPREADER_H
