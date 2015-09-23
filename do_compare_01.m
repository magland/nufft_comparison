function do_compare_01

rng(2);

nufft_comparison_setup;

eps=1e-4;

% Create input data
%E=create_3d_radial_example(100,100,100);
E=create_random_sampling_example(1e6);
%E=create_single_point_example([pi/5+pi,pi/7,pi/9]);
%E=create_single_point_example([0,0,0]);
%E=create_single_point_example([pi/3,pi/6,pi/7]);
xyz=cat(2,E.x,E.y,E.z);
d=E.d;
N1=200; N2=200; N3=200;

% gold standard
opts_blocknufft_gold.eps=1e-10;
opts_blocknufft_gold.K1=80; opts_blocknufft_gold.K2=80; opts_blocknufft_gold.K3=80;
opts_blocknufft_gold.num_threads=8;
opts_blocknufft_gold.kernel_type=2; % 1 -> gaussian, 2 -> KB

%nufft3d1f90
opts_nufft3d1f90.eps=eps * 10; %Note: jfm multiplied by 10 here

% blocknufft
opts_blocknufft.eps=eps;
opts_blocknufft.K1=1000; opts_blocknufft.K2=1000; opts_blocknufft.K3=1000;
opts_blocknufft.num_threads=1;
opts_blocknufft.kernel_type=2; % 1 -> Gaussian, 2 -> KB

% blocknufft_gaussian
opts_blocknufft_gaussian=opts_blocknufft;
opts_blocknufft_gaussian.kernel_type=1; % 1 -> Gaussian, 2 -> KB

%blocknufft with blocking
opts_blocknufft_blocking=opts_blocknufft;
opts_blocknufft_blocking.K1=80; opts_blocknufft_blocking.K2=80; opts_blocknufft_blocking.K3=80;

%blocknufft with multiple threads
opts_blocknufft_multithread=opts_blocknufft_blocking;
opts_blocknufft_multithread.num_threads=1;

%fessler
opts_fessler.oversamp=4;
opts_fessler.spreadR=7;

%nfft
opts_nfft_m1.m=1;
opts_nfft_m2.m=2;
opts_nfft_m3.m=3;
opts_nfft_m4.m=4;
opts_nfft_m5.m=5;
opts_nfft_m6.m=6;
opts_nfft_m7.m=7;

% Uncomment the algorithms you want to test/compare
% Gold standard uses eps=1e-10
algorithms={
    struct('name','gold standard','alg_init',@alg_trivial_init,'alg_run',@alg_blocknufft,'algopts',opts_blocknufft_gold)
    struct('name','nufft fortran','alg_init',@alg_trivial_init,'alg_run',@alg_nufft3d1f90,'algopts',opts_nufft3d1f90)
    %struct('name','blocknufft-g','alg_init',@alg_trivial_init,'alg_run',@alg_blocknufft,'algopts',opts_blocknufft_gaussian)
    %struct('name','blocknufft','alg_init',@alg_trivial_init,'alg_run',@alg_blocknufft,'algopts',opts_blocknufft)
    %struct('name','blocknufft-b','alg_init',@alg_trivial_init,'alg_run',@alg_blocknufft,'algopts',opts_blocknufft_blocking)
    struct('name','blocknufft-mb','alg_init',@alg_trivial_init,'alg_run',@alg_blocknufft,'algopts',opts_blocknufft_multithread)
    %struct('name','nufft Fessler','alg_init',@alg_fessler_init,'alg_run',@alg_fessler_run,'algopts',opts_fessler)
    %struct('name','nfft','alg_init',@alg_nfft_init,'alg_run',@alg_nfft_run,'algopts',opts_nfft_m1)
    %struct('name','nfft','alg_init',@alg_nfft_init,'alg_run',@alg_nfft_run,'algopts',opts_nfft_m2)
    %struct('name','nfft','alg_init',@alg_nfft_init,'alg_run',@alg_nfft_run,'algopts',opts_nfft_m3)
    %struct('name','nfft','alg_init',@alg_nfft_init,'alg_run',@alg_nfft_run,'algopts',opts_nfft_m4)
    %struct('name','nfft','alg_init',@alg_nfft_init,'alg_run',@alg_nfft_run,'algopts',opts_nfft_m5)
    %struct('name','nfft','alg_init',@alg_nfft_init,'alg_run',@alg_nfft_run,'algopts',opts_nfft_m6)
    %struct('name','nfft','alg_init',@alg_nfft_init,'alg_run',@alg_nfft_run,'algopts',opts_nfft_m7)
};

results={};

for j=1:length(algorithms)
    alg_init=algorithms{j}.alg_init;
    alg_run=algorithms{j}.alg_run;
    algopts=algorithms{j}.algopts;
    fprintf('Initializing %s...\n',algorithms{j}.name);
    mem_before=get_used_memory();
    tic;
    obj=alg_init(N1,N2,N3,xyz,algopts);
    results{j}.init_time=toc;
    mem_after=get_used_memory();
    results{j}.memory_used=(mem_after-mem_before)/1e6;
    fprintf('Running %s...\n',algorithms{j}.name);
    tic;
    results{j}.output=alg_run(N1,N2,N3,xyz,d,obj,algopts);
    results{j}.run_time=toc;
    fprintf('Run time: %g\n',results{j}.run_time);
end

%[GX,GY,GZ]=ndgrid((0:N1-1)-floor(N1/2),(0:N2-1)-floor(N2/2),(0:N3-1)-floor(N3/2));
%X=exp(i*(GX*xyz0(1)+GY*xyz0(1)+GZ*xyz0(1)))/1;

%results{1}.output(N1/2+1:N1/2+4,N2/2+1:N2/2+4,N3/2+1)
%X(N1/2+1:N1/2+4,N2/2+1:N2/2+4,N3/2+1)

MM=length(algorithms);
max_diffs=zeros(MM,MM);
avg_diffs=zeros(MM,MM);
for j1=1:MM
    X1=results{j1}.output;
    for j2=1:MM
        X2=results{j2}.output;
        m0=(max(abs(X1(:)))+max(abs(X2(:))))/2;
        max_diffs(j1,j2)=max(abs(X1(:)-X2(:)))/m0;
        avg_diffs(j1,j2)=mean(abs(X1(:)-X2(:)))/m0;
    end;
end;

fprintf('Max. differences:\n');
fprintf('%15s ','');
for j2=1:MM
    fprintf('%15s ',algorithms{j2}.name);
end;
fprintf('\n');
for j1=1:MM
    fprintf('%15s ',algorithms{j1}.name);
    for j2=1:MM
        fprintf('%15g ',max_diffs(j1,j2));
    end;
    fprintf('\n');
end;
fprintf('\n');
fprintf('Avg. differences:\n');
fprintf('%15s ','');
for j2=1:MM
    fprintf('%15s ',algorithms{j2}.name);
end;
fprintf('\n');
for j1=1:MM
    fprintf('%15s ',algorithms{j1}.name);
    for j2=1:MM
        fprintf('%15g ',avg_diffs(j1,j2));
    end;
    fprintf('\n');
end;
fprintf('\n');
fprintf('%15s %15s %15s %15s\n','Algorithm','Init time (s)','Run time (s)','RAM (GB)');
for j1=1:MM
    fprintf('%15s %15.3f %15.3f %15.3f\n',algorithms{j1}.name,results{j1}.init_time,results{j1}.run_time,results{j1}.memory_used);
end;
fprintf('\n');

end

function obj=alg_trivial_init(N1,N2,N3,xyz,opts)
obj=[];
end

function obj=alg_fessler_init(N1,N2,N3,xyz,opts)

spreadR=opts.spreadR;
oversamp=opts.oversamp;
obj=nufft_fessler_init(xyz,[N1,N2,N3],[spreadR,spreadR,spreadR],[oversamp*N1,oversamp*N2,oversamp*N3],[N1/2,N2/2,N3/2]);

end

function X=alg_fessler_run(N1,N2,N3,xyz,d,obj,opts)

%fprintf('\n');
%for j=1:20
%fprintf(' . ');
X=nufft_fessler_adj(d,obj);
%end;
%fprintf('\n');
X=X/length(d);

end

function plan=alg_nfft_init(N1,N2,N3,xyz,opts)

M=size(xyz,1);
N=[N1;N2;N3];
%plan=nfft(3,N,M); 
n=2^(ceil(log(max(N))/log(2))+1);
plan=nfft(3,N,M,n,n,n,opts.m,bitor(PRE_PHI_HUT,bitor(PRE_PSI,NFFT_OMP_BLOCKWISE_ADJOINT)),FFTW_MEASURE); % use of nfft_init_guru

plan.x=xyz/(2*pi); % set nodes in plan
nfft_precompute_psi(plan); % precomputations

end

function X=alg_nfft_run(N1,N2,N3,xyz,d,obj,opts)

M=size(xyz,1);
obj.f=d;
for k=1:10
nfft_adjoint(obj);
end;
X=reshape(obj.fhat,[N1,N2,N3])/M;

end

function X=alg_blocknufft(N1,N2,N3,xyz,d,obj,opts)

X=blocknufft3d(N1,N2,N3,xyz,d,opts.eps,opts.K1,opts.K2,opts.K3,opts.num_threads,opts.kernel_type);

% P=3;
% M=length(d);
% tmp=randi(P,M,1);
% X=zeros(N1,N2,N3);
% for j=1:P
%     inds1=find(tmp==j);
%     X=X+blocknufft3d(N1,N2,N3,xyz(inds1,:),d(inds1,1),opts.eps,opts.K1,opts.K2,opts.K3,opts.num_threads)*length(inds1);
% end;
% X=X/M;

end

function X=alg_nufft3d1f90(N1,N2,N3,xyz,d,obj,opts)

X=nufft3d1f90(xyz(:,1),xyz(:,2),xyz(:,3),d,0,opts.eps,N1,N2,N3);

end

function ret=get_used_memory

[r,w] = unix('free | grep Mem');
stats = str2double(regexp(w, '[0-9]*', 'match')); 
ret=stats(2);

end


