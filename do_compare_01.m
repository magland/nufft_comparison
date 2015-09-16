function do_compare_01

nufft_comparison_setup;

opts_blocknufft_gold.eps=1e-10;
opts_blocknufft_gold.K1=1000; opts_blocknufft_gold.K2=1000; opts_blocknufft_gold.K3=1000;
opts_blocknufft_gold.num_threads=1;

opts_blocknufft.eps=1e-3;
opts_blocknufft.K1=1000; opts_blocknufft.K2=1000; opts_blocknufft.K3=1000;
opts_blocknufft.num_threads=1;

opts_blocknufft_blocking=opts_blocknufft;
opts_blocknufft.K1=50; opts_blocknufft_blocking.K2=100; opts_blocknufft_blocking.K3=100;

opts_blocknufft_multithread=opts_blocknufft_blocking;
opts_blocknufft_multithread.num_threads=6;

opts_fessler.oversamp=2;
opts_fessler.spreadR=6;

algorithms={
    struct('alg_init',@alg_trivial_init,'alg_run',@alg_blocknufft,'algopts',opts_blocknufft_gold)
    struct('alg_init',@alg_trivial_init,'alg_run',@alg_blocknufft,'algopts',opts_blocknufft)
    struct('alg_init',@alg_trivial_init,'alg_run',@alg_blocknufft,'algopts',opts_blocknufft_blocking)
    struct('alg_init',@alg_trivial_init,'alg_run',@alg_blocknufft,'algopts',opts_blocknufft_multithread)
    struct('alg_init',@alg_trivial_init,'alg_run',@alg_nufft3d1f90,'algopts',opts_nufft3d1f90)
    struct('alg_init',@alg_fessler_init,'alg_run',@alg_fessler_run,'algopts',opts_fessler)
};

E=create_3d_radial_example(100,30,100);
%E=create_single_point_example([pi/5,pi/7,pi/9]);
xyz=cat(2,E.x,E.y,E.z);
d=E.d;
N1=100; N2=100; N3=100;

results={};

for j=1:length(algorithms)
    alg_init=algorithms{j}.alg_init;
    alg_run=algorithms{j}.alg_run;
    algopts=algorithms{j}.algopts;
    tic;
    obj=alg_init(N1,N2,N3,xyz,algopts);
    results{j}.init_time=toc;
    tic;
    results{j}.output=alg_run(N1,N2,N3,xyz,d,obj,algopts);
    results{j}.run_time=toc;
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
        max_diffs(j1,j2)=max(abs(X1(:)-X2(:)));
        avg_diffs(j1,j2)=mean(abs(X1(:)-X2(:)));
    end;
end;

fprintf('Max. differences:\n');
for j1=1:MM
    for j2=1:MM
        fprintf('%15g ',max_diffs(j1,j2));
    end;
    fprintf('\n');
end;
fprintf('\n');
fprintf('Avg. differences:\n');
for j1=1:MM
    for j2=1:MM
        fprintf('%15g ',avg_diffs(j1,j2));
    end;
    fprintf('\n');
end;
fprintf('\n');
fprintf('Elapsed times (init / run) (seconds):\n');
for j1=1:MM
    fprintf('%d: %.3f/ %.3f\n',j1,results{j1}.init_time,results{j1}.run_time);
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

X=nufft_fessler_adj(d,obj);
X=X/length(d);

end

function X=alg_blocknufft(N1,N2,N3,xyz,d,obj,opts)

X=blocknufft3d(N1,N2,N3,xyz,d,opts.eps,opts.K1,opts.K2,opts.K3,opts.num_threads);

end

function X=alg_nufft3d1f90(N1,N2,N3,xyz,d,obj,opts)

X=nufft3d1f90(xyz(:,1),xyz(:,2),xyz(:,3),d,0,opts.eps,N1,N2,N3);

end
