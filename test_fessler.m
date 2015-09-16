close all;

E=create_3d_radial_example(50,50,50);
M=E.M;
xyz=cat(2,E.x,E.y,E.z);
d=E.d;
eps=1e-5;
K1=50000; K2=50000; K3=50000;
num_threads=1;

tic;
st=nufft_fessler_init(xyz,[N1,N2,N3],[6,6,6],[2*N1,2*N2,2*N3],[N1/2,N2/2,N3/2]);
fprintf('Fessler: time for init: %g\n',toc);
tic;
A_fessler=nufft_fessler_adj(d,st);
A_fessler=A_fessler/M;
fprintf('Fessler: time for nufft: %g\n',toc);
