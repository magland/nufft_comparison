path0=fileparts(which(mfilename));

% We don't want to set up the entire irt, to reduce risk of name conflicts
addpath([path0,'/nufft_fessler/nufft']);
addpath([path0,'/nufft_fessler/utilities']);
addpath([path0,'/nufft_fessler/systems']);
addpath([path0,'/nufft_fessler/systems/arch']);
addpath([path0,'/blocknufft']);
addpath([path0,'/nufft-1.33-mcwrap']);
addpath([path0,'/../nfft-3.3.0/matlab/nfft']);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Instructions for installing nfft
% Download current source from here: https://www-user.tu-chemnitz.de/~potts/nfft/download.php
% Extract to /../nfft-3.3.0
% cd to nfft-3.3.0 and run
% > ./configure --with-matlab-arch=glnxa64 --with-matlab=/home/magland/MATLAB/ --enable-all --enable-openmp
% Change /home/magland/MATLAB/ to appropriate matlab path and change glnxa64 to appropriate architecture
% > make
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%