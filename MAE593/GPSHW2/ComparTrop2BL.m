clear all

N=2400;

load estENUP1(nodelay).mat
ENUP1BL=estenuP1;
load estENUP1T.mat
ENUP1T=estenuP1;
load estENUP2(nodelay).mat
ENUP2BL=estenuP2;
load estENUP2T.mat
ENUP2T=estenuP2;
load clkBiasP1(nodelay).mat
clkBiasP1BL=clockBiasEstP1;
load clkBiasP1T.mat
clkBiasP1T=clockBiasEstP1;
load clkBiasP2(nodelay).mat
clkBiasP2BL=clockBiasEstP1;
load clkBiasP2T.mat
clkBiasP2T=clockBiasEstP1;
load TruthENU.mat
load TruthClkBias.mat

RMSEenuBL=zeros(3,N);
RMSEenuT=zeros(3,N);

%Find RMSE w/o troposphere
for i=1:N    
RMSEenu(1,i)=sqrt(mean((ENUP1BL(1,i)-truthenu(1,i)).^2));
RMSEenu(2,i)=sqrt(mean((ENUP1BL(2,i)-truthenu(2,i)).^2));
RMSEenu(3,i)=sqrt(mean((ENUP1BL(3,i)-truthenu(3,i)).^2));
RMSEclkB(i)=sqrt(mean((clkBiasP1BL(i)-truthClockBias(i)).^2));
RMSE3d(:,i)=sqrt(mean((ENUP1BL(:,i)-truthenu(:,i)).^2));
end

%find RMSE including troposphere
for i=1:N    
RMSEenuT(1,i)=sqrt(mean((ENUP1T(1,i)-truthenu(1,i)).^2));
RMSEenuT(2,i)=sqrt(mean((ENUP1T(2,i)-truthenu(2,i)).^2));
RMSEenuT(3,i)=sqrt(mean((ENUP1T(3,i)-truthenu(3,i)).^2));
RMSEclkBT(i)=sqrt(mean((clkBiasP1T(i)-truthClockBias(i)).^2));
RMSE3dT(:,i)=sqrt(mean((ENUP1T(:,i)-truthenu(:,i)).^2));
end

subplot(321)
plot(1:N,RMSEenu)
subplot(323)
plot(1:N,RMSE3d)
subplot(325)
plot(1:N,RMSEclkB)

subplot(322)
plot(1:N,RMSEenuT)
subplot(324)
plot(1:N,RMSE3dT)
subplot(326)
plot(1:N,RMSEclkBT)


