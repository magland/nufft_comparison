eps=1e-10;
nj=100; nk=136;
ms=30; mt=20; mu=10;
iflag=1;

% 1D type 1
    xj=rand(nj,1); cj=rand(nj,1)+i*rand(nj,1);
    [fk,ier]=nufft1d1f90(xj,cj,iflag,eps,ms);
    fk_compare=zeros(ms,1);
    for k1=1:ms
        fk_compare(k1)=1/nj*sum(cj.*exp(i*(k1-1-ms/2)*xj));
    end;
fprintf('1D type 1 max err = %g\n',max(abs(fk-fk_compare)));

% 1D type 2
    xj=rand(nj,1); fk=rand(ms,1)+i*rand(ms,1);
    [cj,ier]=nufft1d2f90(xj,iflag,eps,fk);
    cj_compare=zeros(nj,1);
    for j=1:nj
        cj_compare(j)=sum(fk.*exp(i*((1:ms)'-1-ms/2)*xj(j)));
    end;
fprintf('1D type 2 max err = %g\n',max(abs(cj-cj_compare)));

% 1D type 3
    xj=rand(nj,1); cj=rand(nj,1)+i*rand(nj,1); sk=rand(nk,1);
    [fk,ier]=nufft1d3f90(xj,cj,iflag,eps,sk);
    fk_compare=zeros(nk,1);
    for k=1:nk
        fk_compare(k)=sum(cj.*exp(i*sk(k)*xj));
    end;
fprintf('1D type 3 max err = %g\n',max(abs(fk-fk_compare)));

% 2D type 1
    xj=rand(nj,1); yj=rand(nj,1); cj=rand(nj,1)+i*rand(nj,1);
    [fk,ier]=nufft2d1f90(xj,yj,cj,iflag,eps,ms,mt);
    fk_compare=zeros(ms,mt);
    for k1=1:ms
    for k2=1:mt
        fk_compare(k1,k2)=1/nj*sum(cj.*exp(i*(k1-1-ms/2)*xj).*exp(i*(k2-1-mt/2)*yj));
    end;
    end;
fprintf('2D type 1 max err = %g\n',max(abs(fk(:)-fk_compare(:))));

% 2D type 2
    xj=rand(nj,1); yj=rand(nj,1); fk=rand(ms,mt)+i*rand(ms,mt);
    [cj,ier]=nufft2d2f90(xj,yj,iflag,eps,fk);
    cj_compare=zeros(nj,1);
    for j=1:nj
        for k1=1:ms
            cj_compare(j)=cj_compare(j)+sum(squeeze(fk(k1,:)).*exp(i*(k1-1-ms/2)*xj(j)).*exp(i*((1:mt)-1-mt/2)*yj(j)));
        end;
    end;
fprintf('2D type 2 max err = %g\n',max(abs(cj-cj_compare)));

% 2D type 3
    xj=rand(nj,1); yj=rand(nj,1); cj=rand(nj,1)+i*rand(nj,1); sk=rand(nk,1); tk=rand(nk,1);
    [fk,ier]=nufft2d3f90(xj,yj,cj,iflag,eps,sk,tk);
    fk_compare=zeros(nk,1);
    for k=1:nk
        fk_compare(k)=sum(cj.*exp(i*sk(k)*xj).*exp(i*tk(k)*yj));
    end;
fprintf('2D type 3 max err = %g\n',max(abs(fk-fk_compare)));

% 3D type 1
    xj=rand(nj,1); yj=rand(nj,1); zj=rand(nj,1); cj=rand(nj,1)+i*rand(nj,1);
    [fk,ier]=nufft3d1f90(xj,yj,zj,cj,iflag,eps,ms,mt,mu);
    fk_compare=zeros(ms,mt,mu);
    for k1=1:ms
    for k2=1:mt
    for k3=1:mu
        fk_compare(k1,k2,k3)=1/nj*sum(cj.*exp(i*(k1-1-ms/2)*xj).*exp(i*(k2-1-mt/2)*yj).*exp(i*(k3-1-mu/2)*zj));
    end;
    end;
    end;
fprintf('3D type 1 max err = %g\n',max(abs(fk(:)-fk_compare(:))));

% 3D type 2
    xj=rand(nj,1); yj=rand(nj,1); zj=rand(nj,1); fk=rand(ms,mt,mu)+i*rand(ms,mt,mu);
    [cj,ier]=nufft3d2f90(xj,yj,zj,iflag,eps,fk);
    cj_compare=zeros(nj,1);
    for j=1:nj
        for k1=1:ms
        for k2=1:mt
            cj_compare(j)=cj_compare(j)+sum(squeeze(fk(k1,k2,:)).*exp(i*(k1-1-ms/2)*xj(j)).*exp(i*(k2-1-mt/2)*yj(j)).*exp(i*((1:mu)'-1-mu/2)*zj(j)));
        end;
        end;
    end;
fprintf('3D type 2 max err = %g\n',max(abs(cj-cj_compare)));

% 3D type 3
    xj=rand(nj,1); yj=rand(nj,1); zj=rand(nj,1); cj=rand(nj,1)+i*rand(nj,1); sk=rand(nk,1); tk=rand(nk,1); uk=rand(nk,1);
    [fk,ier]=nufft3d3f90(xj,yj,zj,cj,iflag,eps,sk,tk,uk);
    fk_compare=zeros(nk,1);
    for k=1:nk
        fk_compare(k)=sum(cj.*exp(i*sk(k)*xj).*exp(i*tk(k)*yj).*exp(i*uk(k)*zj));
    end;
fprintf('3D type 3 max err = %g\n',max(abs(fk-fk_compare)));
