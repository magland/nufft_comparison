function out=create_3d_radial_example(Nr,Ntheta,Nphi)

N1=50; N2=50; N3=50;
Nr=50; Ntheta=50; Nphi=50;
M=Nr*Ntheta*Nphi;
r=(0.5:1:Nr-0.5)/Nr;
theta=(0.5:1:Ntheta-0.5)/Ntheta*2*pi;
phi=(0.5:1:Nphi-0.5)/Nphi*pi-pi/2;
[R,THETA,PHI]=ndgrid(r,theta,phi);
x0=R.*cos(THETA).*cos(PHI); x0=x0(:)*pi+pi;
y0=R.*sin(THETA).*cos(PHI); y0=y0(:)*pi+pi;
z0=R.*sin(PHI); z0=z0(:)*pi+pi;
d=rand(M,1)*2-1;

%incr=25;
%figure; plot3(x0(25:incr:end),y0(25:incr:end),z0(25:incr:end),'.');

out.N1=N1;
out.N2=N2;
out.N3=N3;
out.M=M;
out.x=x0;
out.y=y0;
out.z=z0;
out.d=d;

end