%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Read data from the iRobot Create                                                                 %
% It was modified from the function AllSensorsReadRoomba By; Joel Esposito, US Naval Academy, 2011 %                                                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [BumpRight, BumpLeft, BumpFront, Wall, virtWall, CliffLft, ...
    CliffRgt, CliffFrntLft, CliffFrntRgt, LeftCurrOver, RightCurrOver, ...
    DirtL, DirtR, ButtonPlay, ButtonAdv, Dist, Angle, ...
    Volts, Current, Temp, Charge, Capacity, pCharge]   = Read_Create_2(S_Create);

BumpRight = 0;
BumpLeft = 0;
BumpFront = 0;
Wall = 0;
virtWall = 0;
CliffLft = 0;
CliffRgt = 0;
CliffFrntLft = 0;
CliffFrntRgt = 0;
LeftCurrOver = 0;
RightCurrOver = 0;
DirtL = 0;
DirtR = 0;
ButtonPlay = 0;
ButtonAdv = 0;
Dist = 0;
Angle = 0;
Volts = 0;
Current = 0;
Temp = 0;
Charge = 0;
Capacity = 0;
pCharge = 0;

% IF THERE IS NO DATA?
CreateBytesAvailable=get(S_Create, 'BytesAvailable');
if (CreateBytesAvailable<26)
    CreateBytesAvailable
    return
else
    SerialData=fread(S_Create, CreateBytesAvailable);
    
    BmpWheDrps = dec2bin(SerialData(1),8);  %
    BumpRight = bin2dec(BmpWheDrps(end));  % 0 no bump, 1 bump
    BumpLeft = bin2dec(BmpWheDrps(end-1));
    if BumpRight*BumpLeft==1
        BumpRight =0;
        BumpLeft = 0;
        BumpFront =1;
    else
        BumpFront = 0;
    end
    
    Wall = SerialData(2);  %0 no wall, 1 wall
    CliffLft = SerialData(3); % 0 no cliff, 1 cliff
    CliffFrntLft = SerialData(4);
    CliffFrntRgt = SerialData(5);
    CliffRgt = SerialData(6);
    virtWall = SerialData(7);%0 no wall, 1 wall

    motorCurr = dec2bin( SerialData(8),8 );
    Low1 = motorCurr(end);  % 0 no over curr, 1 over Curr
    Low0 = motorCurr(end-1);  % 0 no over curr, 1 over Curr
    Low2 = motorCurr(end-2);  % 0 no over curr, 1 over Curr
    LeftCurrOver = str2num(motorCurr(end-3));  % 0 no over curr, 1 over Curr
    RightCurrOver = str2num(motorCurr(end-4));  % 0 no over curr, 1 over Curr

    DirtL = SerialData(9);
    DirtR = SerialData(10);
    RemoteCode =  SerialData(11); % coudl be used by remote or to communicate with sendIR command
    
    Buttons = dec2bin(SerialData(12),8);
    ButtonPlay = Buttons(end);
    ButtonAdv = Buttons(end-2);

    Dist = double(typecast(uint16(SerialData(13)*256+SerialData(14)),'int16'))/1000; %fread(serPort, 1, 'int16')/1000; % convert to Meters, signed, average dist wheels traveled since last time called...caps at +/-32
    Angle = double(typecast(uint16(SerialData(15)*256+SerialData(16)),'int16'))*pi/180; % convert to radians, signed,  since last time called, CCW positive

    ChargeState = SerialData(17);
    Volts = (SerialData(18)*256+SerialData(19))/1000;
    Current = double(typecast(uint16(SerialData(20)*256+SerialData(21)),'int16'))/1000;% neg sourcing, pos charging
    Temp  = SerialData(22); 
    Charge =  (SerialData(23)*256+SerialData(24)); % in mAhours
    Capacity =  (SerialData(25)*256+SerialData(26));
    pCharge = Charge/Capacity *100;  % May be inaccurate
end




