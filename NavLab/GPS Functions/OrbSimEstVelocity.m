%clear all data

%define inputs to check error
N=72727;
orgxyz=xyzbl(1,:)
for i=1:N 
    xyz=xyzbl(i,:);
    enubl(:,i)=xyz2enu(xyz,orgxyz);
    
end
figure
plot(1:N,enubl)
for i=1:N
    enu=enubl(:,i);
    xyzpost(:,i)=enu2xyz(enu,orgxyz);

end
figure
plot(1:N,(xyzbl-xyzpost'))

rmse=sqrt(sum((xyzbl-xyzpost').^2)/N);

dTBL=gradient(BLGPST);
dXBL=gradient(BLX);
dYBL=gradient(BLY);
dZBL=gradient(BLZ);

estVxBL=(dXBL./dTBL)*1000;
estVyBL=(dYBL./dTBL)*1000;
estVzBL=(dZBL./dTBL)*1000;



dT=gradient(GPST);
dX=gradient(X);
dY=gradient(Y);
dZ=gradient(Z);

estVx=(dX./dT)*1000;
estVy=(dY./dT)*1000;
estVz=(dZ./dT)*1000;

errT=BLGPST-GPST;
errXpos=X-BLX;
errYpos=Y-BLY;
errZpos=Z-BLZ;

figure 
plot(BLGPST(1:25),Xpos(1:25))

% figure
% plot(GPST,errXpos,GPST,errYpos)
% 
% figure
% plot(BLGPST,errZpos)

errVx=estVx-estVxBL;
errVy=estVy-estVyBL;
errVz=estVz-estVzBL;

% figure
% plot(BLGPST,errVx,BLGPST,errVy)
% xlabel('GPST')
% ylabel('error(m/s)')
% title('velocity error in x and y(June 21 Continuous)')
% legend('Vx','Vy')
% 
% figure
% plot(BLGPST,errVz)
% xlabel('GPST')
% ylabel('error(m/s)')
% title('velocity error in z(June 21 Continuous)')
% legend('Vz')
% 
% figure
% plot(BLGPST,estVxBL,BLGPST,estVyBL)
% xlabel('GPST')
% ylabel('velocity(m/s)')
% title('Baseline velocity in x and y')
% legend('VxBL','VyBL')
% 
% figure
% plot(BLGPST,estVzBL)
% xlabel('GPST')
% ylabel('velocity(m/s)')
% title('Baseline velocity in z')
% legend('VzBL')
% 
% 
% figure
% plot(BLGPST,estVz)
% xlabel('GPST')
% ylabel('velocity(m/s)')
% title('velocity in z(june 21 Continuous)')
% legend('Vz')
% 
% figure
% plot(BLGPST,estVx,BLGPST,estVy)
% xlabel('GPST')
% ylabel('velocity(m/s)')
% title('velocity in x and y (june 21 continuous')
% legend('Vx','Vy')

