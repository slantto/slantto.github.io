clc
clear
close all

load('DPStream');
for k=1:length(Stream)
    Check=mod(sum(Stream(k,1:9)),8);
   
    
    if Stream(k,1)==uint8('A') && Stream(k,2)==uint8('Z') && Check==Stream(k,10)
        Counter=Stream(k,3);
        Elapse=typecast(uint8(Stream(k,4:5)),'int16');
        DCh1=typecast(uint8(Stream(k,6:7)),'int16');
        DCh2=typecast(uint8(Stream(k,8:9)),'int16');
        PacketRx(k,:)=[Counter,Elapse,DCh1,DCh2]; 
    else
        disp('Corruption');
    end
end
figure;plot(PacketRx(:,1));
title('Counter');xlabel('packet number');ylabel('Counter value');
xlim([0,1100]);ylim([0,260]);
figure;plot(PacketRx(:,2));
title('Elapsed Time to assemble packet(seconds)');xlabel('packet number');ylabel('time(seconds)');
xlim([0,1100]);ylim([0,105]);
figure;semilogx(PacketRx(:,3));
title('Data from channels 1');xlabel('packet number');ylabel('Data');
xlim([0,1100]);ylim([-1.1,1.1]);
figure;semilogx(PacketRx(:,4));
title('Data from channels 2');xlabel('packet number');ylabel('Data');
xlim([0,1100]);ylim([-1.1,1.1]);