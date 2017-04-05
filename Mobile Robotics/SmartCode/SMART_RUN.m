%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is the main code for runing the SMART Robot                  %
% Author: Yu Gu                                                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
close all

% Definitions
Ts_Desired=0.1;           % Desired sampling time
Ts=0.1;                   % sampling time is 0.1 second. It can be reduced 
                          % slightly to offset other overhead in the loop
Tend=60;                  % Was 60 seconds;
Total_Steps=Tend/Ts_Desired;    % The total number of time steps;
Create_Full_Speed=0.2;      % The highest speed the robot can travel. (Max is 0.5m/s)

% Magnetometer Calibration Data
Mag_A=[  2.3750     0.2485      -0.2296
         0          2.6714      -0.1862
         0          0           2.5061];        % estimated shape of the soft iron effect
Mag_c=[ 0.0067;     0.2667;     0.0473];        % estimated center of the hard iron effect

% Rate Gyro Biases
P_Bias=0;
Q_Bias=0;
R_Bias=0;
% P_Bias=-0.0019;
% Q_Bias= 0.0053;
% R_Bias=-0.0149;

% Define the main Data Structure
SD= struct( 'Index',zeros(1,Total_Steps),...            % SD stands for SMART Data, which stores all the robot data
            'Time', zeros(1,Total_Steps),...            % The actual time at each time step (s) 
            'Time_Diff', zeros(1,Total_Steps),...       % The time difference between two step (s) 
            'Delay',zeros(1,Total_Steps),...            % Delay needed between each time step. It indicate how much time avaliable for other compuations. 
            'LaserScan', zeros(640,Total_Steps),...     % Laser Scan Data from Kinect (mm)             
            'Logger_Counter', zeros(1,Total_Steps),...  % The Counter number for the SMART data logger 
            'Ax',zeros(1,Total_Steps),...               % Acceleration along the x-Axis (g)
            'Ay',zeros(1,Total_Steps),...               % Acceleration along the y-Axis (g)
            'Az',zeros(1,Total_Steps),...               % Acceleration along the z-Axis (g)
            'P',zeros(1,Total_Steps),...                % Roll Rate (deg/s)
            'Q',zeros(1,Total_Steps),...                % Pitch Rate (deg/s)
            'R',zeros(1,Total_Steps),...                % Yaw Rate (deg/s)
            'Mx',zeros(1,Total_Steps),...               % Magnetic strength along the x-axis (G)
            'My',zeros(1,Total_Steps),...               % Magnetic strength along the y-axis (G)
            'Mz',zeros(1,Total_Steps),...               % Magnetic strength along the z-axis (G)
            'IMU_T',zeros(1,Total_Steps),...            % IMU internal temperacture (C)
            'Roll',zeros(1,Total_Steps),...             % Roll Angle (rad)
            'Pitch',zeros(1,Total_Steps),...            % Pitch Angle (rad)
            'Yaw',zeros(1,Total_Steps),...              % Yaw Angle (rad) 
            'Mag_Heading',zeros(1,Total_Steps),...      % Magnetic Heading (rad) 
            'X',zeros(1,Total_Steps),...                % Robot X Position (m) 
            'Y',zeros(1,Total_Steps),...                % Robot Y Position (m) 
            'RF_F',zeros(1,Total_Steps),...             % Front Range Finder (mm)
            'RF_FL',zeros(1,Total_Steps),...            % Front-Left Range Finder (mm)
            'RF_L',zeros(1,Total_Steps),...             % Left Range Finder (mm)
            'RF_B',zeros(1,Total_Steps),...             % Back Range Finder (mm)
            'RF_R',zeros(1,Total_Steps),...             % Right Range Finder (mm)
            'RF_FR',zeros(1,Total_Steps),...            % Front Right Range Finder (mm)
            'Wall', zeros(1,Total_Steps),...            % Wall sensor of Create (0/1)
            'VirtWall', zeros(1,Total_Steps),...        % Detect the Virtual Wall (0/1)
            'Dist', zeros(1,Total_Steps),...            % Distance Traveled Since Last Call (m)
            'TotalDist', zeros(1,Total_Steps),...       % Total Distance Traveled(m)
            'Angle', zeros(1,Total_Steps),...           % Angle Traveled Since Last Call (rad)
            'TotalAngle', zeros(1,Total_Steps),...      % Total Angle Traveled (rad)
            'CreateVolts', zeros(1,Total_Steps),...     % Voltage of the Create Robot (rad)
            'CreateCurrent', zeros(1,Total_Steps));     % Current of the Create Robot (rad)
   
% Initialize the Serial Port for the Data Logger
S_Logger=Init_Logger('2');  % 6 on another computer
% Initialize the Serial Port for iRobot Create
S_Create=RoombaInit(1);     % 5 on another computer


% %Initialize the Kinect device
% vid = videoinput('kinect', 2, 'Depth_640x480');
% src = getselectedsource(vid);
% set(vid, 'FramesPerTrigger',1);
% set(vid, 'TriggerRepeat',Inf);
% triggerconfig(vid,'manual');
% % Set Region of Interest (ROI)
% vid.ROIPosition = [0 230 640 20];
% start(vid);

flushinput(S_Logger);       % Flush the data logger serial port
flushinput(S_Create);       % Flush the iRobot Create serial port
fwrite(S_Create, [142 0]);  % Request all sensor data from Create
BeepRoomba(S_Create);       % Make a Beeping Sound

% Starting the main loop
for i=1:Total_Steps
    tic
    SD.Index(i)=i;
    
% Use the Code Below if You Want to use the Kinect    
%     trigger(vid);             % Trigger the Kinect to send a depth frame
%     Data=getsnapshot(vid);    % Get the Kinect depth frame
%     flushdata(vid);           % Clean up the memory for storing the image data
%     SD.LaserScan(:,i)=median(Data);
%     for j=1:638
%         A=SD.LaserScan(j,i);
%         B=SD.LaserScan(j+1,i);
%         C=SD.LaserScan(j+2,i);
%         Spike(j+1)=abs((A-B)+(C-B));
%     end
%     plot(SD.LaserScan(:,i));
%     hold on
%     plot(Spike,'r')
%     hold off

    % Acquire data from the sensor interface board
    [SD.Logger_Counter(i), SD.Ax(i), SD.Ay(i), SD.Az(i), Raw_P, Raw_Q, Raw_R, Raw_Mx, Raw_My, Raw_Mz, SD.IMU_T(i), A2D_Ch1, A2D_Ch2, SD.RF_F(i), SD.RF_FL(i), SD.RF_L(i), SD.RF_B(i), SD.RF_R(i), SD.RF_FR(i)] = Read_Logger_2(S_Logger); 
    SD.P(i)=Raw_P-P_Bias;    SD.Q(i)=Raw_Q-Q_Bias;    SD.R(i)=Raw_R-R_Bias;   % Calibrate the gyro data
    temp=Mag_A*[Raw_Mx-Mag_c(1); Raw_My-Mag_c(2); Raw_Mz-Mag_c(3)];           % Magnetometer Raw Data Correction
    SD.Mx(i)=temp(1); SD.My(i)=temp(2); SD.Mz(i)=temp(3);

    % Acquire the data from the iRobot Create
    [BumpRight,BumpLeft,BumpFront,SD.Wall(i),SD.VirtWall(i),CliffLeft,CliffRight,CliffFrontLeft,CliffFrontRight,LeftCurrOver,RightCurrOver,DirtL,DirtR,ButtonPlay,ButtonAv,SD.Dist(i),SD.Angle(i),SD.CreateVolts(i),SD.CreateCurrent(i),Temp,Charge,Capacity,pCharge]=Read_Create_2(S_Create);
    Time=clock;                     % Mark the current time;
    SD.Time(i)=Time(6);             % Store the seconds;
    if i==1
        SD.Roll(i)=atan2(SD.Ay(i), SD.Az(i));                                 % Calculate the Roll angle based on the gravity vector
        SD.Pitch(i)=atan(-SD.Ax(i)/(SD.Ay(i)*sin(SD.Roll(i))+SD.Az(i)*cos(SD.Roll(i))));    % Calculate the Roll angle based on the pitch vector
        SD.Mag_Heading(i)=atan2(-SD.My(i), SD.Mx(i));                         % The initial 2D magnetic heading of the robot
        SD.Yaw(i)=0;                                                          % Set the current yaw angle as zero
        Attitude_P=zeros(3,3);                                                % Initialization the error covariance matrix for attitude estimation 
        SD.X(i)=0;                                                            % Initial Robot X Position
        SD.Y(i)=0;                                                            % Inital Robot Y Position
    else                                                                      % If i>1
        SD.Time_Diff(i)=SD.Time(i)-SD.Time(i-1);                              % Calculate the time difference between steps
        if SD.Time_Diff(i)<0
            SD.Time_Diff(i)=SD.Time_Diff(i)+60;                               % Compensate for the minute change
        end
        SD.TotalDist(i)=SD.TotalDist(i-1)+SD.Dist(i);                         % Calculate the total traveled distance based on the encoder reading
        SD.TotalAngle(i)=SD.TotalAngle(i-1)+SD.Angle(i);                      % Calculate the total traveled angle based on the encoder reading
        
        SD.Mag_Heading(i)=atan2(-SD.My(i), SD.Mx(i));                         % The 2D magnetic heading of the robot

        [SD.Roll(i), SD.Pitch(i), SD.Yaw(i), Attitude_P] = Attitude_Estimation(SD.Mag_Heading(1), SD.Roll(i-1), SD.Pitch(i-1), SD.Yaw(i-1), SD.Time_Diff(i), Attitude_P, SD.Ax(i), SD.Ay(i), SD.Az(i), SD.P(i), SD.Q(i), SD.R(i), SD.Mx(i), SD.My(i), SD.Mz(i));     % Perform the attitude estimation 
        SD.X(i)=SD.X(i-1)+SD.Dist(i)*sin(SD.Yaw(i)-SD.Yaw(1));                % Dead Reconing for X position
        SD.Y(i)=SD.Y(i-1)+SD.Dist(i)*cos(SD.Yaw(i)-SD.Yaw(1));                % Dead Reconing for X position
    end
    
    flushinput(S_Logger);       % Flush the data logger serial port
    flushinput(S_Create);       % Flush the iRobot Create serial port
    fwrite(S_Create, [142 0]);  % Request all sensor data from Create

    
    %% Put your custom control functions here
    %----------------------------------------------------------
    % Use the following function for the robot wheel control:
    % SetDriveWheelsSMART(S_Create, rightWheelVel, leftWheelVel, SD.CliffLeft(i),SD.CliffRight(i),SD.CliffFrontLeft(i),SD.CliffFrontRight(i));


    % Collision Avoidance Example(Set Tend=60s)
    if SD.RF_F(i)<500
        SetDriveWheelsSMART(S_Create, -Create_Full_Speed*.4, -Create_Full_Speed*.6, CliffLeft,CliffRight,CliffFrontLeft,CliffFrontRight,BumpRight,BumpLeft,BumpFront);
    elseif SD.RF_B(i)<300
        SetDriveWheelsSMART(S_Create, Create_Full_Speed*.6, Create_Full_Speed*.4, CliffLeft,CliffRight,CliffFrontLeft,CliffFrontRight,BumpRight,BumpLeft,BumpFront);
 %   elseif SD.RF_FL(i)<300
 %       SetDriveWheelsSMART(S_Create, Create_Full_Speed*.5, -Create_Full_Speed*.5, CliffLeft,CliffRight,CliffFrontLeft,CliffFrontRight,BumpRight,BumpLeft,BumpFront);
 %   elseif SD.RF_FR(i)<300
 %       SetDriveWheelsSMART(S_Create, Create_Full_Speed*.5, -Create_Full_Speed*.5, CliffLeft,CliffRight,CliffFrontLeft,CliffFrontRight,BumpRight,BumpLeft,BumpFront);
    elseif SD.RF_L(i)<500
        SetDriveWheelsSMART(S_Create, -Create_Full_Speed*.5, Create_Full_Speed*0.5, CliffLeft,CliffRight,CliffFrontLeft,CliffFrontRight,BumpRight,BumpLeft,BumpFront);
    elseif SD.RF_R(i)<500
        SetDriveWheelsSMART(S_Create, Create_Full_Speed*0.5, -Create_Full_Speed*.5, CliffLeft,CliffRight,CliffFrontLeft,CliffFrontRight,BumpRight,BumpLeft,BumpFront);
    else
        SetDriveWheelsSMART(S_Create, Create_Full_Speed, Create_Full_Speed, CliffLeft,CliffRight,CliffFrontLeft,CliffFrontRight,BumpRight,BumpLeft,BumpFront);
    end

    %----------------------------------------------------------
    %% End of the Custom Control Code

    % Calculate the time left in the constant interval
    SD.Delay(i)=Ts-toc;         
    if SD.Delay(i)>0
        pause(SD.Delay(i));     % Kill the remaining time
    end
end
Total_Elapse_Time=SD.Time(Total_Steps)-SD.Time(1)  % Calcualte the total elapse time, not counting the minutes
SetDriveWheelsSMART(S_Create, 0, 0, CliffLeft,CliffRight,CliffFrontLeft,CliffFrontRight,BumpRight,BumpLeft,BumpFront);       % Stop the wheels
BeepRoomba(S_Create);       % Make a Beeping Sound

% Properly close Kinect if it was used.
% stop(vid);
% flushdata(vid);
% delete(vid);

% Properly close the serial ports
delete(S_Logger)
clear S_Logger  
delete(S_Create)
clear S_Create  

save('SMART_DATA.mat', 'SD');       % Save all the collected data to a .mat file
SMART_PLOT;                         % Plot all the robot data