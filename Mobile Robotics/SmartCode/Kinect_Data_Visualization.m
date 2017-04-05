%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is the code to visualize the Kinect data                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
close all

%Initialize the Kinect device
%vid = videoinput('kinect', 1, 'RGB_640x480'); %the value 1 indictes the color sensor
vid = videoinput('kinect', 2, 'Depth_640x480'); %the value 2 indictes the depth sensor
src = getselectedsource(vid);
set(vid, 'FramesPerTrigger',1);
set(vid, 'TriggerRepeat',Inf);
triggerconfig(vid,'manual');
% Set Region of Interest (ROI), Comment out this line if you want to see
% the full view
%vid.ROIPosition = [0 230 640 20]; % [Xoffset Yoffset Width Height]

% Initialize the program
Ts=0.0945;                   % sampling time is 0.1 second. It is reduced 
                             % slightly to offset other overhead in the loop

start(vid);

clock
for i=1:200
    tic
    trigger(vid);           % Trigger the Kinect to send a depth frame
    Data=getsnapshot(vid);  % Get the Kinect depth frame
    flushdata(vid);         % Clean up the memory for storing the image data
    
    imaqmontage(Data);      % Show the collected Data
    
    Time_Remain=Ts-toc;     % Calculate the time left in the constant interval
    if Time_Remain>0
        pause(Time_Remain); % Pause the remaining time
    end
end
clock

%Properly close the Kinect. If not done properly, you would have to restart
%the computer.
stop(vid);
flushdata(vid);
delete(vid);