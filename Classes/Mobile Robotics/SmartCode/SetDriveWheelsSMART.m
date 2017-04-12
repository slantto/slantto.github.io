%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Control iRobot Create while checking the cliff and bump sensors                                  %
% It was modified from the function AllSensorsReadRoomba By; Joel Esposito, US Naval Academy, 2011 %                                                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [] = SetDriveWheelsSMART(serPort, rightWheelVel, leftWheelVel, CliffLft, CliffRgt, CliffFrntLft, CliffFrntRgt,BumpRight,BumpLeft,BumpFront)

rightWheelVel = min( max(1000* rightWheelVel, -500) , 500); % Limit the left wheel speed
leftWheelVel = min( max(1000* leftWheelVel, -500) , 500);   % Limit the right wheel speed
Right_temp=typecast(int16(rightWheelVel),'uint8');          % Seperate the command into two bytes for transmission
Left_temp=typecast(int16(leftWheelVel),'uint8');
if CliffLft==0 && CliffRgt==0 && CliffFrntLft==0 && CliffFrntRgt==0 && BumpRight==0 && BumpLeft==0 && BumpFront==0% If there is no cliff and wheels are not stuck
    fwrite(serPort, [145 Right_temp(2) Right_temp(1) Left_temp(2) Left_temp(1)]);   % Drive as commanded speed
else
    fwrite(serPort, [145 0 0 0 0]);      % stop the wheels
end