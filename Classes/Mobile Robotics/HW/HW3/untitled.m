mesDistRaw=(3.3*(double(S_rawdata)/4096))/0.252;

%% Two Point Calibration
halfmetercalavg=mean(mesDistRaw(1,1:100));
fivemetercalavg=mean(mesDistRaw(1,901:1000));
Cslope=(0.5-5)/(halfmetercalavg-fivemetercalavg);
Coffset=0.5-(halfmetercalavg*Cslope);
mesDistCaltwopoint=Coffset+(Cslope*mesDistRaw); 

%% Multipoint Calibration
p=polyfit(1:1000,mesDistRaw,1);
mesDistCalmultipoint=((mesDistRaw*p(1))+p(2));

plot(1:1000,mesDistRaw,1:1000,mesDistCaltwopoint,1:1000,mesDistCalmultipoint);
legend('Raw','2-point Calibrated','Multipoint Calibrated');
title('Sonar measured distance');grid on
xlabel('index');ylabel('measurment');