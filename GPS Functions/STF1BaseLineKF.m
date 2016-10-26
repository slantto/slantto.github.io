clear all

load('OrbBaseline.mat')

load BaselineXYZT.txt;
xyzBL=1000*BaselineXYZT(:,2:4);
xyzBL=xyzBL';

%Setting Constants
N=length(RangeL1); %number of epochs
c=299792458; %speed of light, m/s
nomXYZ=xyzorg;
clockBiasNom=0;


smooth=input('Carrier Phase Smoothing?(1 for yes, 0 for no)');
velocity=input('given prRate?(1 for yes, 0 for no)');
trop=input('Include Troposphere delay(1 for yes 0 for no)');
if trop==1
    Po=1013;
    To=288.15;
    eo=12.8;
    %     Po=input('what is your total pressure?(mbar)');
    %     To=input('what is your temperature?(kelvin)');
    %     eo=input('what is your partial pressure do to water?(mbar)');
else
    Po=0;
    To=0;
    eo=0;
end

if velocity==1
    %state-vector  x=[delta-x, delta-y, delta-z, clock bias, Vx, Vy, Vz,clock drift]
    
    %assume delta is zero intially
    x=[0 0 0 0 0 0 0 0]';
    
    % initiallize uncertainty, 100m at xyz
    P=[100^2 0 0 0 0 0 0 0 ;
        0 100^2 0 0 0 0 0 0 ;
        0 0 100^2 0 0 0 0 0 ;
        0 0 0 10000^2 0 0 0 0 ;
        0 0 0 0 100^2 0 0 0;
        0 0 0 0 0 100^2 0 0;
        0 0 0 0 0 0 100^2 0;
        0 0 0 0 0 0 0 100^2];
    
    % State transition matrix
    %user is stationary
    F=[1 0 0 0 1 0 0 0;
        0 1 0 0 0 1 0 0;
        0 0 1 0 0 0 1 0;
        0 0 0 1 0 0 0 1;
        0 0 0 0 1 0 0 0;
        0 0 0 0 0 1 0 0;
        0 0 0 0 0 0 1 0;
        0 0 0 0 0 0 0 1];
    
    % process noise
    Qpos=00
    Qvel=00
    Q=[ Qpos^2 0 0 0 0 0 0 0;
        0 Qpos^2 0 0 0 0 0 0;
        0 0 Qpos^2 0 0 0 0 0;
        0 0 0 100 0 0 0 0;
        0 0 0 0 Qvel^2 0 0 0;
        0 0 0 0 0 Qvel^2 0 0;
        0 0 0 0 0 0 Qvel^2 0;
        0 0 0 0 0 0 0 100];
    
elseif velocity==0
    
    %state-vector  x=[delta-x, delta-y, delta-z, clock bias, clock drift]
    
    %assume delta is zero intially
    x=[0 0 0 0 0]';
    
    % initiallize uncertainty, 100m at xyz
    P=[100^2 0 0 0 0;
        0 100^2 0 0 0;
        0 0 100^2 0 0;
        0 0 0 10000^2 0
        0 0 0 0 10^2];
    
    % State transition matrix
    %user is stationary
    F=[1 0 0 0 0;
        0 1 0 0 0;
        0 0 1 0 0;
        0 0 0 1 1;
        0 0 0 0 1];
    
    % process noise
    Qpos=0
    Q=[ Qpos^2 0 0 0 0;
        0 Qpos^2 0 0 0;
        0 0 Qpos^2 0 0;
        0 0 0 100 0;
        0 0 0 0 100];
    
end

%use Ionosphereic free data
prDataIF=(2.546*RangeL1)-(1.546*RangeL2);
if smooth==1
    M=100;
    prSmooth=carrierSmooth(PhaseL1,PhaseL2,M,prDataIF);
    prDataIF=prSmooth;
end
if velocity==1
    prRateL2(:,1600)=prRateL1(:,1600);
    prRateIF=(2.546*prRateL1)-(1.546*prRateL2);
end
llh=xyz2llh(nomXYZ);
nSat=NumerOfSatsUsed;
for i=1:N
    satsXYZ=Sats(1:NumerOfSatsUsed(i),:,i);
    
    % step #1 predict state
    x=F*x;
    
    % step #2 predict error-covarince
    P=F*P*F'+Q;
    
    % form observation matrix
    n=length(x);
    if velocity==1
        m=2*nSat(i);
    elseif velocity==0
        m=nSat(i);
    end
    H=zeros(m,n);
    
    prComputed=zeros(nSat(i),1);
    
    
    for j=1:nSat(i)
        prComputed(j)=norm(satsXYZ(j,:)-nomXYZ)+clockBiasNom*c;
        if velocity==1
            prSquigDot(j)=prRateIF(j)-(satsVxVyVz(j,:,i)*((satsXYZ(j,:,i)-nomXYZ)/norm(satsXYZ(j,:,i)-nomXYZ))');
        end
        H(j,1:4)=[(satsXYZ(j,:)-nomXYZ)/norm(satsXYZ(j,:)-nomXYZ), 1];
    end
    
    if velocity==1
        H(((m/2)+1):m,5:8)=-1*H(1:(m/2),1:4);
    end
    
    % form meaurement error covariance ( could do elevation dependent weighting here)
    R=(2^2)*eye(m);
%     for j=1:nSat(i)
%         Satenu(j,:)=xyz2enu(satsXYZ(j,:,i),nomXYZ);
%         sinel(j,:)=(Satenu(j,3))/(norm(Satenu(j,:)));
%         R(j,j)=R(j,j)*(1/sinel(j));
%         el(j,i)=asind(sinel(j,:));
%         mel(j)=1.001/(sqrt(0.002001+((sind(el(j,i)))^2)));%found on navipedia
%         if trop==1
%             prComputed(j)=prComputed(j)+mel(j)*TropSaastamoinen(llh,Po,To,eo);
%         end
%     end
    
    
    % step #3 compute Kalman Gain
    K=P*H'*inv(H*P*H'+R);
    
    % step #4 update state
    % measurement vector
    if velocity==0
        z=prComputed'-prDataIF(i,1:nSat(i)); %delta-rho
        y=H*x;
        x=x+K*(z'-y);
    elseif velocity==1
        z(1:nSat(i))=prComputed-prDataIF(1:nSat(i),i); %delta-rho
        z((nSat(i)+1):(2*nSat(i)))=prSquigDot;
        y=H*x;
        x=x+K*(z'-y);
        
    end
    
    % step # 5 update measurment error covariance
    P=(eye(length(x))-K*H)*P;
    
    
    % step 6 save estimate and move to next step
    xyzKF(i,1:3)=nomXYZ'+x(1:3);
    clockBiasKF(i)=clockBiasNom'+(x(4));
    KFclkDrift(i)=x(5);
    if velocity==1
    VxyzKF(i,1:3)=x(5:7);
    end
    enuTruth(i,:)=xyz2enu(xyzBL(:,i),nomXYZ);
    enuKF(i,:)=xyz2enu(xyzKF(i,1:3),nomXYZ);
    KF_3DErr(i)=norm(enuKF(i,:)-enuTruth(i,:));
    %KF_clkBiasErr(i)=(clockBiasKF(i)-truthClockBias(i))/c;
    llh=xyz2llh(xyzKF(i,:));
    
end
% figure
% plot(KF_3DErr)
% figure
% plot(KF_clkBiasErr/1000)
figure
plot(enuKF-enuTruth)
title('KF ENU Error')
xlabel('Epoch')
ylabel('error(m)')
legend('e','n','u')
% subplot(312)
% plot(KF_clkBiasErr)
% title('KF Clock Bias Error')
% xlabel('Epoch')
% ylabel('error(m)')
% subplot(212)
% plot(KFclkDrift)
% title('KF Clock Drift')
% xlabel('Epoch')
% ylabel('seconds')
RMSEenu(1)=sqrt(mean((enuKF(:,1)-enuTruth(:,1)).^2));
RMSEenu(2)=sqrt(mean((enuKF(:,2)-enuTruth(:,2)).^2));
RMSEenu(3)=sqrt(mean((enuKF(:,3)-enuTruth(:,3)).^2));
figure
plot(KF_3DErr)
title('KF 3D Error')
xlabel('Epoch')
ylabel('error(m)')
if velocity==1
figure
plot(VxyzKF)
title('KF Velocity Estimation(in XYZ)')
xlabel('Epoch')
ylabel('Velocity(m/s)')
legend('Vx','Vy','Vz')
end
figure
plot(xyzKF-xyzBL')