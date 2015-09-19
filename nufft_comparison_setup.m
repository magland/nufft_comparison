path0=fileparts(which(mfilename));

% We don't want to set up the entire irt, to reduce risk of name conflicts
addpath([path0,'/nufft_fessler/nufft']);
addpath([path0,'/nufft_fessler/utilities']);
addpath([path0,'/nufft_fessler/systems']);
addpath([path0,'/nufft_fessler/systems/arch']);
addpath([path0,'/blocknufft']);
addpath([path0,'/nufft-1.33-mcwrap']);
addpath([path0,'/../nfft-3.3.0/matlab/nfft']);