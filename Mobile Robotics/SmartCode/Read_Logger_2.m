%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Read the SMART Robot Sensor Interface Board                       %
% Author: Yu Gu                                                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [counter, IMU_Ax, IMU_Ay, IMU_Az, IMU_P, IMU_Q, IMU_R, IMU_Mx, IMU_My, IMU_Mz, IMU_T, A2D_Ch1, A2D_Ch2, A2D_F, A2D_FL, A2D_L, A2D_B, A2D_R, A2D_FR] = Read_Logger_2(S_Logger)
counter=0;
IMU_Ax=0;
IMU_Ay=0;
IMU_Az=0;
IMU_P=0;
IMU_Q=0;
IMU_R=0;
IMU_Mx=0;
IMU_My=0;
IMU_Mz=0;
IMU_T=0;
A2D_Ch1=0;
A2D_Ch2=0;
A2D_F=0;
A2D_FL=0;
A2D_L=0;
A2D_B=0;
A2D_R=0; 
A2D_FR=0;

% IF THERE IS NO DATA?
LoggerBytesAvailable=get(S_Logger, 'BytesAvailable');
if (LoggerBytesAvailable<50)
    LoggerBytesAvailable
    return
else
    SerialData=fread(S_Logger, LoggerBytesAvailable);
    for i=1:(LoggerBytesAvailable-50)
        if SerialData(i)==65            % Check the first Header
            if SerialData(i+1)==90        % Check the second Header
                counter=SerialData(i+2);
                temp=SerialData(i+3)*256+SerialData(i+4);
                IMU_Ax = 0.0033*double(typecast(uint16(temp),'int16'));

                temp=SerialData(i+5)*256+SerialData(i+6);
                IMU_Ay = 0.0033*double(typecast(uint16(temp),'int16'));    % Negative sign because of the fliped IMU
             
                temp=SerialData(i+7)*256+SerialData(i+8);
                IMU_Az = 0.0033*double(typecast(uint16(temp),'int16'));
             
                temp=SerialData(i+9)*256+SerialData(i+10);
                IMU_P = deg2rad(0.05*double(typecast(uint16(temp),'int16')));   % output the gyro in rad/sec
 
                temp=SerialData(i+11)*256+SerialData(i+12);
                IMU_Q = deg2rad(0.05*double(typecast(uint16(temp),'int16')));

                temp=SerialData(i+13)*256+SerialData(i+14);
                IMU_R = deg2rad(0.05*double(typecast(uint16(temp),'int16')));

                temp=SerialData(i+15)*256+SerialData(i+16);
                IMU_Mx = 0.0005*double(typecast(uint16(temp),'int16'));

                temp=SerialData(i+17)*256+SerialData(i+18);
                IMU_My = 0.0005*double(typecast(uint16(temp),'int16'));

                temp=SerialData(i+19)*256+SerialData(i+20);
                IMU_Mz = 0.0005*double(typecast(uint16(temp),'int16'));
             
                temp=SerialData(i+21)*256+SerialData(i+22);
                IMU_T = 0.14*double(typecast(uint16(temp),'int16'))-25;

                temp=SerialData(i+23)*256+SerialData(i+24);
                Voltage=3.3*(temp/4096);
                A2D_Ch1=Voltage;                   % Convert A/D counts to voltage

                temp=SerialData(i+25)*256+SerialData(i+26);
                Voltage=3.3*(temp/4096);
                A2D_Ch2=Voltage;

                temp=SerialData(i+27)*256+SerialData(i+28);
                Voltage=3.3*(temp/4096);
                A2D_F=Voltage/0.000252;              % At 3.3 V input, 6.4mv/in lead to 0.252mv/mm
             
                temp=SerialData(i+29)*256+SerialData(i+30);
                Voltage=3.3*(temp/4096);
                A2D_FL=Voltage/0.000252;              

                temp=SerialData(i+31)*256+SerialData(i+32);
                Voltage=3.3*(temp/4096);
                A2D_L=Voltage/0.000252;              

                temp=SerialData(i+33)*256+SerialData(i+34);
                Voltage=3.3*(temp/4096);
                A2D_B=Voltage/0.000252;              

                temp=SerialData(i+35)*256+SerialData(i+36);
                Voltage=3.3*(temp/4096);
                A2D_R=Voltage/0.000252;              
             
                temp=SerialData(i+37)*256+SerialData(i+38);
                Voltage=3.3*(temp/4096);
                A2D_FR=Voltage/0.000252;              
             
                return
            end
        end
    end
end