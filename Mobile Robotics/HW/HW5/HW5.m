%Homework 5
%Sean Lantto
clear
close all
clc
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
    forceTru(t) = 5*sin(Time(t));
    %AccelTru(t) = (forceTru(t)/Mass)-(k*PosTru(t)/Mass)-(c*VelTru(t)/Mass);
    if t > 1
        AccelTru(t) = (-k*PosTru(t)/Mass)-(c*VelTru(t)/Mass)+(forceTru(t)/Mass);
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

figure; plot(Time,PosTru,Time,PosMes,Time,PosEst);
grid on; legend('True position','Measured position','Estimated position');
xlabel('time(Seconds)'); ylabel('Position(meters)');
