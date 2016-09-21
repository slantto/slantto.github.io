%% Comparison of KF vs LLS for stand alone positioning with no atmospheric modeling
%% Lecture 11 MAE 593I
load dataSet1.mat % static data
C=299792458;
% iterative
i=1;
[xyzIT(i,:),clockBiasIT(i)]=LLSGPS(nomXYZ,clockBiasNom,satsXYZ(1:nSat(i),:,i),prData(1:nSat(i),i), nSat(i),0,0,0,0);
enuTruth(1,:)=xyz2enu(truthXYZ(:,1),nomXYZ);
 enuLLS(i,:)=xyz2enu(xyzIT(i,:),nomXYZ);
for i=2:2400
    %  W=diag(sin(Elev(
    [xyzIT(i,:),clockBiasIT(i)]=LLSGPS(xyzIT(i-1,:),clockBiasIT(i-1),satsXYZ(1:nSat(i),:,i),prData(1:nSat(i),i), nSat(i),0,0,0,0);
    enuLLS(i,:)=xyz2enu(xyzIT(i,:),nomXYZ);
    enuTruth(i,:)=xyz2enu(truthXYZ(:,i),nomXYZ);
    LLS_3DErr(i)=norm(enuLLS(i,:)-enuTruth(i,:));
end

%%% GPS Static Kalman Filter with PR Data %%%%%

%% step #1

%% state-vector  x=[delta-x, delta-y, delta-z, delta-b, delta-xb-drift]
% takes advantage of clock biases that grown linearly ( lots of receiver
% clocks)

%% initization assume delta is zero intially
x=[0 0 0 0 0]';

%% initiallize uncertainty
% assume nominal is up to 1000 meters off, ms for bias and micro-second for
% drift
P=[2000^2 0 0 0 0;
    0 1200^2 0 0 0;
    0 0  2000^2 0 0;
    0 0 0  300000^2 0;
    0 0 0 0   100^2;]

%% State transition matrix
% we know the user is not moving!!

PHI=[1 0 0 0 0;
    0 1 0 0 0;
    0 0 1 0 0;
    0 0 0 1 1; % clock bias is the only dynamic state ( clock bias=clock bias + drift )
    0 0 0 0 1;];


%% process noise
% our model is really good for user position because we no the user is
% static, sum uncertainty in clock model tho
Qpos=0
Q=[ Qpos^2 0 0 0 0 ;
    0 Qpos^2 0 0 0 ;
    0 0 Qpos^2 0 0 ;
    0 0 0 50 0;
    0 0 0 0  50]; % 50 meters is really small in seconds

%% measurement noise
% this depends on how many measurements we have, so need to do in loop

for i=1:2400
    
    %% step #1 predict state
    x=PHI*x;
    %% step #2 predict error-covarince
    P=PHI*P*PHI'+Q;
    
    %% form observation matrix ( this is same model that we used for LLS)
    %% initialize sizes
    m=nSat(i);
    H=zeros(m,5);
    prComputed=zeros(m,1);
    for j=1:nSat(i)
        prComputed(j)=norm(satsXYZ(j,:,i)-nomXYZ)+clockBiasNom*C;
        H(j,1:4)=[(satsXYZ(j,:,i)-nomXYZ)/norm(satsXYZ(j,:,i)-nomXYZ), 1];
        
    end
    
    %% form meaurement error covariance ( could do elevation dependent weighting here)
    R=(2.5^2)*eye(m);
    
    %% step #3 compute Kalman Gain
    K=P*H'*inv(H*P*H'+R);
    
    %% step #4 update state
    % measurement vector
    z=prComputed-prData(1:nSat(i),i);
    y=H*x;
    x=x+K*(z-y);
    
    %% step # 5 update measurment error covariance
    P=(eye(5)-K*H)*P;
    
    %% step 6 save estimate and move to next step
    xyzKF(i,1:3)=nomXYZ'+x(1:3);
    enuKF(i,:)=xyz2enu(xyzKF(i,1:3),nomXYZ);
    KF_3DErr(i)=norm(enuKF(i,:)-enuTruth(i,:));
end

close all
figure
top=[ 12 12 25];
bottom=[-5 -5 -5];
label=['E(m)';'N(m)';'V(m)']
for i=1:3
subplot(3,1,i),plot(enuLLS(:,i)-enuTruth(:,i),'--')
hold on
plot(enuKF(:,i)-enuTruth(:,i),'r','LineWidth',1)
xlim([500 1500])

legend('LLS','KF')
ylabel(label(i,:))
grid
end
xlabel('Time into run (s)')

sprintf('KF Mean: %f STD %f',mean(KF_3DErr),std(KF_3DErr))
sprintf('LLS Mean: %f STD %f',mean(LLS_3DErr),std(LLS_3DErr))


