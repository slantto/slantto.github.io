%Homework 5
%Sean Lantto
clear
close all
clc

%problem 1
%% Generate Truth and measurements
Mass = 1; %kg
k = 5; %N/m
c = 1; %Ns/m
T_S = 0.1;
T = 1000; %Run Time in seconds (# of iterations)

Time = zeros(T,1);
AccelTru = zeros(T,1);
VelTru = zeros(T,1);
PosTru = zeros(T,1);
ForceMes = AccelTru;
PosMes = PosTru;
VelEst = VelTru;
PosEst = PosTru;

for t = 1 : T
    Time(t) = t*T_S;
    forceTru(t) = 5*(sin(Time(t)));
    %AccelTru(t) = (forceTru(t)/Mass)-(k*PosTru(t)/Mass)-(c*VelTru(t)/Mass);
    if t > 1
        AccelTru(t) = ((-k*PosTru(t-1))/Mass)-((c*VelTru(t-1))/Mass)+((forceTru(t))/Mass);
        VelTru(t) = VelTru(t-1) + AccelTru(t) * T_S;
        PosTru(t) = PosTru(t-1) + VelTru(t) * T_S;
    end
    PosMes(t) = PosTru(t) + normrnd(0,0.2);
    ForceMes(t) = (forceTru(t) + normrnd(0,0.05));    
end
%% Kalman Filter
X = zeros(T,2)'; %Initialize State Vector
P = eye(2); %Initialize covariance matrix

A = [1,T_S;-k*T_S/Mass,1-c*T_S/Mass]; 
B = [0;T_S/Mass];
C = [1,0];
Q = [0,0;0,0.05^2]*(T_S^2); %Proccess Noise
R = 0.4; %Measurement Noise

for n = 2 : T
    %% Prediction Step
    X(:,n) = A*X(:,n-1) + B*ForceMes(n);
    P = A*P*A' + Q;
    
    %% Measurement Step
    K = P*C'*inv(C*P*C' + R);
    X(:,n) = X(:,n) + K*(PosMes(n) - C*X(:,n));
    P = (eye(2) - K*C)*P;
    PosEst(n) = X(1,n);
    VelEst(n) = X(2,n);
end

figure; plot(Time,PosTru);hold on;plot(Time,PosMes);plot(Time,PosEst,'LineWidth',3);
grid on; legend('True position','Measured position','Estimated position');
xlabel('time(Seconds)'); ylabel('Position(meters)');title('Mass/spring/damper KF results');

%%Problem 2

%% Create truth and mesaurements
m=1;%kg
l=0.5;%m
c=0.5;%coefficient of friction
g=9.81;%m/sec^2
T_s=0.1;%10 Hz sampling
T=1000;%number of samples
ThetaDDT = zeros(T,1);
ThetaDT = zeros(T,1);
ThetaT = zeros(T,1);
ThetaT(1)=1.6;
HorzDistMes = ThetaDT;
ThetaDE = ThetaDT;
ThetaE = ThetaDT;
ThetaDDT(1)=((-g/l)*sin(ThetaT(1)))-((c/(m*l))*ThetaDT(1));
   

for t=2:T
    ThetaDDT(t)=((-g/l)*sin(ThetaT(t-1)))-((c/(m*l))*ThetaDT(t-1));
    ThetaDT(t)=ThetaDDT(t)*T_S+ThetaDT(t-1);
    ThetaT(t)=ThetaDT(t)*T_s+ThetaT(t-1);
    
    HorzDistMes(t)=(l*sin(ThetaT(t)))+normrnd(0,0.5);
end

figure;plot(Time,ThetaT,Time,ThetaDT,Time,HorzDistMes);
legend('True theta','True theta dot','measured distance(m)');
xlabel('time(seconds)');title('Model results')

%% Initialize Extended Kalman filter
X = zeros(T,2)'; %Initialize State Vector
X(:,1)=[1.6;0];
P = eye(2); %Initialize covariance matrix
%F=[1,T_s;(-T_s*(g/l)*cos(X(1,1))),(1-T_s*(c/(m*l)))];
%H=[l*cos(X(1,1)),0];
Q=[0.0001,0;0,0.0001];
R=0.25;

for k=2:T
    %Predict State
    X(:,k)=[X(1,k-1)+(T_s*X(2,k-1));X(2,k-1)-(T_s*(g/l)*sin(X(1,k-1)))-(T_s*(c/(m*l))*X(2,k-1))];
    %Calculate prediction jacobian and Predict Error covariance
    F=[1,T_s;(-T_s*(g/l)*cos(X(1,k-1))),(1-T_s*(c/(m*l)))];
    P=F*P*F'+Q;
    thetapredict(k)=X(1,k);
    thetadotpred(k)=X(2,k);
   %Innovation
    r=HorzDistMes(k)-(l*sin(X(1,k)));
    %Calculate measurment jacobian and innovation covariance
    H=[l*cos(X(1,k)),0];
    S=H*P*H'+R;
    %Calculate Kalman Gain
    K=P*H'*inv(S);
    %Posterior state estimate
    X(:,k)=X(:,k)+(K*r);
    %posterior error covariance
    P=(eye(2)-(K*H))*P;
    
    ThetaE(k)=X(1,k);
    ThetaDE(k)=X(2,k);
end

figure;plot(Time,ThetaT,Time,ThetaE,Time,thetapredict);
legend('true','est','predict');xlabel('Time (seconds)');ylabel('Angle (Radians)');title('Non-Forced EKF Results')