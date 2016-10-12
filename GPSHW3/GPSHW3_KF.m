%state-vector  x=[delta-x, delta-y, delta-z, clock bias, clock drift, Vx, Vy, Vz, Troposphere]

%assume delta is zero intially
x=[0 0 0 0 0 0 0 0 0]';

% initiallize uncertainty
P=[1000^2 0 0 0 0 0 0 0 0;
    0 1000^2 0 0 0 0 0 0 0;
    0 0 1000^2 0 0 0 0 0 0;
    0 0 0 100000^2 0 0 0 0 0;
    0 0 0 0 100^2 0 0 0 0;
    0 0 0 0 0 1^2 0 0 0;
    0 0 0 0 0 0 1^2 0 0;
    0 0 0 0 0 0 0 1^2 0;
    0 0 0 0 0 0 0 0 10^2];

% State transition matrix
PHI=[1 0 0 0 0 0 0 0 0;
    0 1 0 0 0 0 0 0 0;
    0 0 1 0 0 0 0 0 0;
    0 0 0 1 1 0 0 0 0; % clock bias is the only dynamic state ( clock bias=clock bias + drift )
    0 0 0 0 1 0 0 0 0;
    0 0 0 0 0 1 0 0 0;
    0 0 0 0 0 0 1 0 0;
    0 0 0 0 0 0 0 1 0;
    0 0 0 0 0 0 0 0 1];

% process noise
Qpos=0
Qvel=0
Q=[ Qpos^2 0 0 0 0 0 0 0 0 ;
    0 Qpos^2 0 0 0 0 0 0 0;
    0 0 Qpos^2 0 0 0 0 0 0;
    0 0 0 50 0 0 0 0 0;
    0 0 0 0  50 0 0 0 0;
    0 0 0 0 0 Qvel^2 0 0 0;
    0 0 0 0 0 0 Qvel^2 0 0;
    0 0 0 0 0 0 0 Qvel^2 0;
    0 0 0 0 0 0 0 0 100];

% measurement noise

for i=1:2400
    
    % step #1 predict state
    x=PHI*x;
    
    % step #2 predict error-covarince
    P=PHI*P*PHI'+Q;
    
    % form observation matrix
    
    m=nSat(i);
    H=zeros(m,5);
    prComputed=zeros(m,1);
    for j=1:nSat(i)
        prComputed(j)=norm(satsXYZ(j,:,i)-nomXYZ)+clockBiasNom*C;
        H(j,1:4)=[(satsXYZ(j,:,i)-nomXYZ)/norm(satsXYZ(j,:,i)-nomXYZ), 1];
    end
    
    % form meaurement error covariance ( could do elevation dependent weighting here)
    R=(2.5^2)*eye(m);
    
    % step #3 compute Kalman Gain
    K=P*H'*inv(H*P*H'+R);
    
    % step #4 update state
    % measurement vector
    z=prComputed-prData(1:nSat(i),i);
    y=H*x;
    x=x+K*(z-y);
    
    % step # 5 update measurment error covariance
    P=(eye(5)-K*H)*P;
    
    % step 6 save estimate and move to next step
    xyzKF(i,1:3)=nomXYZ'+x(1:3);
    enuKF(i,:)=xyz2enu(xyzKF(i,1:3),nomXYZ);
    KF_3DErr(i)=norm(enuKF(i,:)-enuTruth(i,:));
    
end