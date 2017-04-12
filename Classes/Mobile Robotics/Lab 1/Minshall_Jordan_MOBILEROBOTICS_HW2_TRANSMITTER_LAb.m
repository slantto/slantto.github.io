%Jordan Minshall 
%Transmitter
pt=0.01;
stream = [];
Stream=[];
tic; %Time refrence
count = uint8(0); %Initialize byte counter

for n = 1:1024
pause(pt); %time pause

Head0 = uint8(65); %Headers
Head1 = uint8(90);

Byte0 = Head0;
Byte1 = Head1;



Byte2 = count; %ByteCounter

if count == 255     %Byte Counter reset 
    count = uint8(0);
else
end

t = toc;    %take time

WholeT = t*500; %Amplify time by a range usable
Byte34 = typecast(uint16(WholeT),'uint8'); %Convert time to uint16 then split into uint8


%Both Sin functions are aplified by 10000 converted to int16 then split
%into uint8

SINt = sin(t);
Whole56 = SINt * 10000;

Byte56W = int16(Whole56);
Byte56= typecast(Byte56W,'uint8');


SIN2t = sin(20*t);
Whole78 = SIN2t * 10000;

Byte78W = int16(Whole78);
Byte78= typecast(Byte78W,'uint8');


%Sm is the sum for the checksum

Sm=double(Byte0)+double(Byte1)+double(Byte2)+double(Byte34(1))+double(Byte34(2))+double(Byte56(1))+double(Byte56(2))+double(Byte78(1))+double(Byte78(2));
Sm=uint8(mod(uint32(Sm),256));


Byte9 = Sm;
%for the packets and the stream
PKT= [Byte0 Byte1 Byte2 Byte34(1) Byte34(2) Byte56(1) Byte56(2) Byte78(1) Byte78(2) Byte9];
%stream = [stream PKT];
count = count +1;


fwrite(Cereal19,[PKT]);

PKTrecieved = fread(Cereal2);

Stream = [Stream transpose(PKTrecieved)];

end

%fwrite(Cereal19,[stream]);

%Jordan Minshall
%Reciever


Packet = [];
DATA = zeros(4,1024); %Preallocated data matrix
for p = 1:1024 %This counts packets
    
    PackStart = 10*(p-1)+1; %Determines the begging and ending indice of each packet
    PackEnd = 10*p;
    
    if PackStart == 0
        PackStart = 1;
    else
    end
    
    j = 1;
    for P = PackStart:PackEnd
        
        Packet(j) = Stream(P);    %forms the active packet
        j = j +1;
    end
    
    CheckSumCount = 0;
    
    for b = 1:10           %Cycles through the bytes of the active packet
        
        Byte = Packet(b);
        
        DecByte = double(Byte);
        
         switch b
             
             case 1         %Check first header
                if DecByte == 65
                    fprintf('Packet %1.0f Header 1 correct, 65 \n',p)
                else
                    fprintf('ERROR Packet %1.0f Header 1, %3.0f \n',p,DecByte)
                end
                
             case 2          %Check second header
                 
                 if DecByte == 90
                    fprintf('Packet %1.0f Header 2 correct, 90 \n',p)
                 else
                    fprintf('ERROR Packet %1.0f Header 2, %3.0f \n',p,DecByte)
                 end
                
             case 3     %Saves Counter for active packet
                 
                 Counter255 = DecByte;
                 
             case 4     %Saves first time byte of active packet
                 
                 TIME1 = DecByte;
                 
             case 5        %Saves second time byte of active packet
                 
                 TIME2 = DecByte;
                 
             case 6         %saves 1st sin(t) byte of active packet
                 
                  SINt1 = DecByte;
                  
                  
             case 7 %saves 2nd sin(t) byte of active packet
                 
                 SINt2 = DecByte;
                 
             case 8 %saves 1st sin(20t) byte of active packet
                 
                 SIN20t1 = DecByte;
                 
             case 9 %Saves 2nd sin(20t) byte of active packer
             
                  SIN20t2 = DecByte;
                 
             case 10 %Checks checksum
                 
                 if Byte == uint8(mod(uint32(CheckSumCount),256));
                      fprintf('Packet %1.0f CheckSum Correct \n',p)
                 else
                       fprintf('ERROR Packet %1.0f,CheckSum = %3.0f Count = %3.0f \n',p,DecByte,CheckSumCount)
                 end
                 ChkSum = uint8(Byte);
         end
         
         %adds bytes to verify checksum
         CheckSumCount = CheckSumCount + DecByte;
    


    end
    %Combines data into in16
    Time = typecast(uint8([TIME1 TIME2]),'uint16');
    Sint = typecast(uint8([SINt1 SINt2]),'int16');
    Sin20t = typecast(uint8([SIN20t1 SIN20t2]),'int16');
    
    % converts to double then divides this was the error I had been
    % expierencing when we saw each other in the lab I had been dividing an
    % int16 and matlab was dropping the decimals
    Time = double(Time)/500;
    Sint = double(Sint)/10000;
    Sin20t = double(Sin20t)/10000;
 
    %archive packet data so the loop can continue to the next
    
    DATA(1,p) = Counter255;
    DATA(2,p) = Time;
    DATA(3,p) = Sint;
    DATA(4,p) = Sin20t;
    DATA(5,p) = ChkSum;
    DATA(6,p) = p;
                 
end

plot(DATA(2,:),DATA(1,:));
title('Counter vs Time');
xlabel('Time (s)');
ylabel('Count');
 
figure
 
plot(DATA(2,:),DATA(3,:));
title('Sin(t) vs Time');
xlabel('Time (s)');
ylabel('Sin(t)');
 
figure
 
plot(DATA(2,:),DATA(4,:));
title('Sin(20t) vs Time');
xlabel('Time (s)');
ylabel('Sin(20t)');

figure

plot(DATA(6,:),DATA(2,:));
title('Time');
xlabel('Packet count');
ylabel('Time(s)');
 
figure

plot(DATA(6,:),DATA(5,:));
title('Checksum')
xlabel('Packet Number')
ylabel('Checksum')

save('Lab1TxRxPause01')