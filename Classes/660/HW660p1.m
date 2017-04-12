clear all
close all

%Probelm 7-32
% G(s)=K/s(0.1s+1)(s+1)

OLnum=[0 0 0 1];
OLden=[0.1 1.1 1 0];
sys_uncomp=tf(OLnum,OLden);

% Design requirements (achieve at least the indicated values):
Kv=4;   % 1/sec
PM=45;   % deg
GM=8;   % dB
%% Steps 1 and two can be seen on the attached hand work
% Step #1
% Consider this format for the TF of the Lead Compensator:
% Gc=K*(Ts+1)/(alphaTs+1)

% Step #2
% Determine K from the Kv requirement
% See attached
K=2;
%%
% Step #3
% Draw the Bode diagram of G1(s) = K*G(s) and calculate the phase and gain
% margins
OLnum1=K*OLnum;

OLden1=OLden;
sysG1=tf(OLnum1,OLden1);

figure
bode(sysG1)
title('Bode Diagram of G1')
grid
figure
margin(sysG1)

[Gm1,Pm1,Wcg1,Wcp1] = margin(sysG1);
disp('Gain margin of G1 [dB]')
disp(20*log10(Gm1))
disp('Phase margin of G1 [deg]')
disp(Pm1)
%%
% Step #4
% Determine the additional lead phase necessary to meet the PM
% requirement. 

Phim=PM-Pm1+9;  % degrees %allow extra magin to account for gain crossover frequenc
disp('Additional lead phase necessary [deg]:')
disp(Phim)
%%
% Step #5
% Determine alpha 
alpha=(1-sin(Phim*pi/180))/(1+sin(Phim*pi/180));
%%
% Step #6
% Determine T 
% First, determine the gain cross-over frequency of the compensated system
magGC=-20*log10(1/sqrt(alpha));
[mag,phase,w]=bode(sysG1);
for i=1:length(w);
magDB(i)=20*log10(mag(i));
end
wGC=1/interp1(sort(magDB),sort(1./w),magGC);

disp('Cross-over frequency of the compensated system GcG')
disp(wGC)


T=1/(wGC*sqrt(alpha));

% Step #7
Kc=K/alpha;

%%
% Step #8 - Verify Design
% Check the frequency response of the OL compensated system

OLnum_comp=K*conv([0 0 T 1],OLnum);
OLden_comp=conv([0 0 alpha*T 1],OLden);
sys_comp=tf(OLnum_comp,OLden_comp);

OL_num_lead=Kc*[T 1];
OL_den_lead=[alpha*T 1];
sys_lead=tf(OL_num_lead,OL_den_lead);
figure
bode(sysG1,sys_lead,sys_comp,{0.001,100})
legend('G1','Lead Compensator','Compensated System - OL')
grid


figure
bode(sys_comp,{0.001,100})
title('Bode Diagram of the Compensated System GcG')
grid

[mag_comp,phase_comp,w_comp]=bode(sys_comp,{0.001,100});
[mag_uncomp,phase_uncomp,w_uncomp]=bode(sys_uncomp,{0.001,100});
[mag_G1,phase_G1,w_G1]=bode(sysG1,{0.001,100});

for i=1:length(w_comp)
mag_comp1(i)=mag_comp(i); 
mag_uncomp1(i)=mag_uncomp(i);  
mag_G11(i)=mag_G1(i); 
phase_comp1(i)=phase_comp(i); 
phase_uncomp1(i)=phase_uncomp(i);    
phase_G11(i)=phase_G1(i); 
end

figure
plot(log10(w_uncomp),20*log10(mag_uncomp1),'k',log10(w_G1),20*log10(mag_G11),'b',log10(w_comp),20*log10(mag_comp1),'r')
legend('Uncompensated system','G1','Compensated system')
title('Magnitude Bode plot')
xlabel('Log10(Frequency)')
ylabel('Magnitude [dB]')
grid

figure
plot(log10(w_uncomp),phase_uncomp1,'k',log10(w_G1),phase_G11,'b',log10(w_comp),phase_comp1,'r')
legend('Uncompensated system','G1','Compensated system')
title('Phase Bode plot')
xlabel('Log10(Frequency)')
ylabel('Phase [deg]')
grid


[Gm_comp,Pm_comp,Wcg_comp,Wcp_comp] = margin(sys_comp);
disp('Gain margin of compensated system [dB]')
disp(20*log10(Gm_comp))
disp('Phase margin of compensated system [degrees]')
disp(Pm_comp)

% Check the time domain response of the CL compensated system

sys_uncomp_cl=tf(OLnum,OLnum+OLden);
sys_G1_cl=tf(OLnum1,OLnum1+OLden1);
sys_comp_cl=tf(OLnum_comp,OLnum_comp+OLden_comp);

[y1,t1]=step(sys_uncomp_cl);
[y2,t2]=step(sys_G1_cl);
[y3,t3]=step(sys_comp_cl,t1);


figure,plot(t1,y1,'k',t2,y2,'b',t3,y3,'r')
title('Unit Step Response')
legend('Uncompensated System','G1','Compensated System')
xlabel('Time [s]')
grid

% Check the ramp response of the CL compensated system
time=0:0.02:50;
y4=lsim(sys_uncomp_cl,time,time);
y5=lsim(sys_G1_cl,time,time);
y6=lsim(sys_comp_cl,time,time);

figure,plot(time,time,'k--',time,y4,'k',time,y5,'b',time,y6,'r')
title('Response to Ramp Input')
legend('Ramp Input','Uncompensated System','G1','Compensated System')
xlabel('Time [s]')
grid



















