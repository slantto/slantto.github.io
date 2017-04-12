%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EKF for SMART Robot Attitude Estimation                           %
% Author: Yu Gu                                                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Roll, Pitch, Yaw, Attitude_P] = Attitude_Estimation(Initial_Heading, Pre_Roll, Pre_Pitch, Pre_Yaw, Time_Diff, Pre_P, IMU_Ax, IMU_Ay, IMU_Az, IMU_P, IMU_Q, IMU_R, IMU_Mx, IMU_My, IMU_Mz)

Roll_M=atan2(IMU_Ay, IMU_Az);                                 % Calculate the Roll angle based on the gravity vector
Pitch_M=atan(-IMU_Ax/(IMU_Ay*sin(Roll_M)+IMU_Az*cos(Roll_M)));    % Calculate the Roll angle based on the pitch vector
Yaw_M=atan2((IMU_Mz*sin(Roll_M)-IMU_My*cos(Roll_M)),(IMU_Mx*cos(Pitch_M)+IMU_My*sin(Pitch_M)*sin(Roll_M)+IMU_Mz*sin(Pitch_M)*cos(Roll_M)))-Initial_Heading; %3D Magnetic Heading Solution with Tile Compensation
%Yaw=atan2(-Cal_My, Cal_Mx); %2D Magnetic Heading Solution

% Initialize Variables
Ts=Time_Diff;                                                % Sampling time 

Q= [1.1339e-006     0               0
    0               3.1232e-006     0
    0               0               2.9009e-006];        % Process Noise Covariance
R= [0.001            0               0
    0               0.001            0
    0               0               0.1];                % Measurement Noise Covariance
I=eye(3,3);

% Pre Calculate Trigonometric Values
sin_Phi=sin(Pre_Roll);
cos_Phi=cos(Pre_Roll);
tan_Theta=tan(Pre_Pitch);
sec_Theta=sec(Pre_Pitch);

% Step #1: State Prediction (Dead Reconing)
Roll_DR=Pre_Roll+Ts*(IMU_P+IMU_Q*sin_Phi*tan_Theta+IMU_R*cos_Phi*tan_Theta);
Pitch_DR=Pre_Pitch+Ts*(IMU_Q*cos_Phi-IMU_P*sin_Phi);
Yaw_DR=Pre_Yaw+Ts*(IMU_Q*sin_Phi+IMU_R*cos_Phi)*sec_Theta;

% Step #2: Update Error Covariance minus
% Calculate the Jacobian matrix
A_k=[1+Ts*tan_Theta*(IMU_Q*cos_Phi-IMU_R*sin_Phi)   Ts*(sec_Theta^2)*(IMU_Q*sin_Phi+IMU_R*cos_Phi)        0
    -Ts*(IMU_Q*sin_Phi+IMU_R*cos_Phi)               1                                                     0
     Ts*(IMU_Q*cos_Phi-IMU_R*sin_Phi)*sec_Theta     Ts*sec_Theta*tan_Theta*(IMU_Q*sin_Phi+IMU_R*cos_Phi)  1];
       
Pm_k=A_k*Pre_P*A_k'+Q;
    
% Step #3: Update Feedback gain K
K_k=Pm_k*inv(Pm_k+R);
    
% Step #4: Update States:
Total_Acceleration=(IMU_Ax^2+IMU_Ay^2+IMU_Az^2)^.5;       % Calculate the total acceleration
Total_Mag=(IMU_Mx^2+IMU_My^2+IMU_Mz^2)^.5;                % Calculate the total magnetic measurement
if Total_Acceleration<1.05 && Total_Acceleration>0.95;    % only update pitch and roll if the total acceleration is around 1g 
    Roll=Roll_DR+K_k(1,1)*(Roll_M-Roll_DR);
    Pitch=Pitch_DR+K_k(2,2)*(Pitch_M-Pitch_DR);
    Attitude_P=(I-K_k)*Pm_k; 
else  % no update
    Roll=Roll_DR;
    Pitch=Pitch_DR;
    Attitude_P=Pm_k;
end
if Total_Mag<1.1 && Total_Mag>0.9;    % only update yaw if the total normalized magnetific field is around 1
    Yaw_Diff=Yaw_M-Yaw_DR;
    while Yaw_Diff<-pi                % wrap the angle
        Yaw_Diff=Yaw_Diff+2*pi;
    end
    while Yaw_Diff>pi
        Yaw_Diff=Yaw_Diff-2*pi;
    end
    Yaw=Yaw_DR+K_k(3,3)*(Yaw_Diff);   
    
% Step #5: Update Error Covariance
    Attitude_P=(I-K_k)*Pm_k; 
else    % No update
    Yaw=Yaw_DR;
    Attitude_P=Pm_k;
end

