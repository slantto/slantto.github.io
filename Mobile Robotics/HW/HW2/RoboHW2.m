clear
clc

tic
Stream=zeros(1024,10);
for k=1:1024
 
Head1='A';Head1=uint8(Head1);
Head2='Z';Head2=uint8(Head2);
Counter=uint8(mod(k,256)-1);
t=toc;
Elapse=typecast(int16(t),'uint8');
D1=sin(t);
D2=sin(20*t);
DCh1=typecast(int16(D1),'uint8');
DCh2=typecast(int16(D2),'uint8');
sump=sum([Head1, Head2, Counter, Elapse, DCh1, DCh2]);
Check=mod(sump,8);

Packet=[Head1 Head2 Counter Elapse DCh1 DCh2 Check];

Stream(k,:)=Packet;

pause(0.1);

end
save('DPStream','Stream')