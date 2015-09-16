path0=fileparts(which(mfilename));

% We don't want to set up the entire irt, to reduce risk of name conflicts
addpath([path0,'/nufft_fessler/nufft']);
addpath([path0,'/nufft_fessler/utilities']);
addpath([path0,'/nufft_fessler/systems']);
addpath([path0,'/nufft_fessler/systems/arch']);