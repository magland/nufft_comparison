function out=create_random_sampling_example(M)

out.x=rand(M,1)*2*pi-pi;
out.y=rand(M,1)*2*pi-pi;
out.z=rand(M,1)*2*pi-pi;
out.d=rand(M,1)*2-1;

end