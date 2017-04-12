clc
clear
load('OrbBaseline.mat')

%"Truth"
load BaselineXYZT.txt;
xyzBL=1000*BaselineXYZT(:,2:4);
xyzBL=xyzBL';

%using L1 Range data

%Constants
c=299792458; % Speed of Light
fL1=1575.42e6;
lambdaL1=c/fL1;
fL2=1227.6e6;
lambdaL2=c/fL2;
Drate=0.1; %10 Hz data
nomXYZ=xyzorg;
clockBiasNom=0;


%state vector
numStates=3; %solving for positiononly
Epochs=length(RangeL1);
x=zeros(numStates,1);

%assume 1000 meter uncertainty in xyz
Ppos=eye(3)*1000^2;
%Pclk
%Pdrift
P=blkdiag(Ppos);

%Cubesat moving at orbital speeds
Qpos=eye(3)*(1000)^2; %will determine if I should include Drate
%Qclk
%Qdrift
Q=blkdiag(Qpos);

%State Transition Matrix
F=eye(numStates);

for i=1:Epochs
    satsXYZ=Sats(1:NumerOfSatsUsed(i),:,i);
    
    R=(ones(NumerOfSatsUsed(i))+eye(NumerOfSatsUsed(i)))*0.02^2;
    
    %predict state
    x=F*x;
    
    %predict error covariance
    P=F*P*F'+Q;
    
    %Pre allocate sizes
    H=zeros(NumerOfSatsUsed(i),numStates);
    prComputed=zeros(NumerOfSatsUsed(i),1);
    
    %Find PR Computed and fill H marix with unit vectors
    for j=1:NumerOfSatsUsed(i)
        prComputed(j)=norm(satsXYZ(j,:)-nomXYZ)+clockBiasNom*c;
        H(j,1:3)=[(satsXYZ(j,:)-nomXYZ)/norm(satsXYZ(j,:)-nomXYZ)];
    end
    
    %Calculate Kalman Gain
    K=P*H'*inv(H*P*H'+R);
    
    %form measurement vector
    z=prComputed'-RangeL1(i,1:NumerOfSatsUsed(i)); %delta-rho
    
    %Update State
    y=H*x;
    x=x+K*(z'-y);
    
    %Update Covariance
    P=(eye(numStates)-K*H)*P;
    
    xyzKF(i,1:3)=nomXYZ'+x(1:3);
    
end

% RMSExyz(1)=sqrt(mean((xyzKF(:,1)-xyzBL(:,1)).^2));
% RMSExyz(2)=sqrt(mean((xyzKF(:,2)-xyzBL(:,2)).^2));
% RMSExyz(3)=sqrt(mean((xyzKF(:,3)-xyzBL(:,3)).^2));


figure
plot(xyzKF-xyzBL')
title('KF XYZ Error')
xlabel('Epoch')
ylabel('error(m)')
legend('x','y','z')
