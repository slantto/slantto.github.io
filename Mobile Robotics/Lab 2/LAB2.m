% Lab2 
% Sean Lantto
% Sierra Portillo
% Diana L. Quiroz

clear
COM1=RoombaInit(1)
% [BumpRight, BumpLeft, BumpFront, Wall, virtWall, CliffLft, ...
%     CliffRgt, CliffFrntLft, CliffFrntRgt, LeftCurrOver, RightCurrOver, ...
%     DirtL, DirtR, ButtonPlay, ButtonAdv, Dist, Angle, ...
%     Volts, Current, Temp, Charge, Capacity, pCharge]   = AllSensorsReadRoomba(COM1)
% 
% 
% for i=1:200
%     [Volts(i)]=BatteryVoltageRoomba(COM1);
%     [Current(i)]=CurrentTesterRoomba(COM1);
% end
% figure;plot(1:200,Volts)
% xlabel('i');
% ylabel('Voltage');
% 
% figure;plot(1:200,Current)
% xlabel('i');
% ylabel('Current');

SetLEDsRoomba(COM1, 3, 100,100)

BeepRoomba(COM1)

% for i=1:4
% travelDist(COM1, 0.25, 1)
% turnAngle(COM1,0.1,90)
% end
DistTot=0;
AngTot=0;
i=0;
SetDriveWheelsCreate(COM1, 0.5,0.5)
while 1
    i=i+1;
    
    fwrite(COM1,[142 0]); 
    pause(0.01);
    [BumpRight, BumpLeft, BumpFront, Wall, virtWall, CliffLeft, CliffRight, CliffFrontLeft,CliffFrontRight, LeftCurrOver, RightCurrOver, ...
    DirtL, DirtR,ButtonPlay,ButtonAv, Dist, Angle,CreateVolts,CreateCurrent,Temp,Charge,Capacity,pCharge]   = Read_Create_2(COM1)
Dista(i)=Dist;
Anga(i)=Angle;
DistTot=DistTot+Dista(i);
AngTot=AngTot+Anga(i);
if DistTot>=1
    SetDriveWheelsCreate(COM1,0,0)
    SetDriveWheelsCreate(COM1,.1,-.1)
    DistTot=0;
end
if AngTot>=90*pi/180 || AngTot<=-90*pi/180
    SetDriveWheelsCreate(COM1,0,0)
    SetDriveWheelsCreate(COM1,0.5,0.5)
    AngTot=0;
end
if BumpLeft==1||BumpRight==1||BumpFront==1
    SetDriveWheelsCreate(COM1, 0,0);
    SetDriveWheelsCreate(COM1, -.5,-.5);
    pause(0.5)
    SetDriveWheelsCreate(COM1,.1,-.1)
    pause(1)
    SetDriveWheelsCreate(COM1,.5,.5)
    DistTot=0;
    AngTot=0;
end
if CliffFrontLeft==1||CliffFrontRight==1||CliffLeft==1||CliffRight==1
      SetDriveWheelsCreate(COM1, 0,0);
    break
end
Volta(i)=CreateVolts;
Amp(i)=CreateCurrent;
pause(0.05)
end

save('fastexptRun')


X(1,1)=0;
Y(1,1)=0;
D(1,1)=0;
A(1,1)=0;

for jj=2:length(Dista)
    A(:,jj)=A(jj-1)+Anga(jj);
    D(:,jj)=D(jj-1)+Dista(jj);
    X(:,jj)=D(jj)*cos(A(jj))+X(:,jj-1);
    Y(:,jj)=D(jj)*sin(A(jj))+Y(:,jj-1);
end

figure; plot(X,Y);
xlabel('X (meters)');ylabel('Y (meters)');
grid on
title('X and Y travel of Robot')


figure; plot(1:length(Volta),Volta);
ylabel('battery voltage (Volts)');
grid on
title('battery voltage')

figure; plot(1:length(Amp),Amp);
ylabel('battery current draw(Amp)');
grid on
title('battery current draw')