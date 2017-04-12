clear all
close all

% 7-34

% G(s)=1/s(s+1)(s+5)


OLnum=[1];
OLden=[1 6 5 0];
sys_uncomp=tf(OLnum,OLden);


% Design requirements (achieve at least the indicated values):
Kv = 20;   % 1/sec
PM = 60;   % deg
GM = 10;    % dB

% Step #1
% Consider this format for the TF of the Lead-Lag Compensator:
% Gc=Kc*(T1s+1)/(alphaT1s+1)*(T2s+1)/(T2s/alpha+1)

% Step #2
% Determine Kc from the Kv requirement
% Off-line: Kv=20=lim(sGc(s)G(s))
Kc=100;

% Step #3
% Draw the Bode diagram of G1(s) = Kc*G(s) and calculate the phase and gain
% margins
OLnum1=Kc*OLnum;
OLden1=OLden;
sysG1=tf(OLnum1,OLden1);

figure
bode(sysG1)
title('Bode Diagram of G1')
grid
[Gm1,Pm1,Wcg1,Wcp1] = margin(sysG1);
disp('Gain margin of G1')
disp(20*log10(Gm1))
disp('Phase margin of G1')
disp(Pm1)

% Step #4
% Determine the location of the gain cross-over frequency for the compensated system.
% Allow 5-12 degrees extra margin to account for extra lag produced by the compensator.
wGC=Wcg1;   % rad/sec
disp('Cross-over frequency of the compensated system GcG')
disp(wGC)
Phim=PM+5;  % degrees
disp('Additional lead phase necessary:')
disp(Phim)

% Step #5
% Select the corner frequency of the lag compensator coresponding to the zero (1/T2) one decade lower
% than the newly selected gain cross-over frequency
% Determine the lag zero.
w_zero=wGC/10;
T2=1/w_zero;
lag_zero=-1/T2;
disp('Zero of the lag compensator located at:')
disp(lag_zero)

% Step #6
% Determine alpha
alpha=(-1-sind(Phim))/(-1+sind(Phim));
disp('Alpha:')
disp(alpha)

% Step #7
% Determine the pole of the lag compensator
lag_pole=-1/alpha*T2;
disp('Pole of the lag compensator located at:')
disp(lag_pole)

% Step #8
% Determine the zero and pole of the lead compensator

w_zero_lead=0.35;  % rad/sec
lead_zero=-w_zero_lead;
lead_pole=lead_zero*alpha;


% Step #9 Verify design
% Check the frequency response of the OL compensated system
lead_num=[1 -lead_zero];
lead_den=[1 -lead_pole];
lag_num=[1 -lag_zero];
lag_den=[1 -lag_pole];

sys_lead=tf(lead_num,lead_den);
sys_lag=tf(lag_num,lag_den);


OLnum_comp=Kc*conv(conv(lead_num,lag_num),OLnum);
OLden_comp=conv(conv(lead_den,lag_den),OLden);
sys_comp=tf(OLnum_comp,OLden_comp);

figure
bode(sys_comp,{0.001,100})
title('Bode Plots of Compensated System - OL')
grid

figure
bode(sys_comp,sys_lead,sys_lag,{0.001,1000})
title('Bode Plots of Compensated System - OL')
grid


[mag_comp,phase_comp,w_comp]=bode(sys_comp,{0.001,100});


[Gm_comp,Pm_comp,Wcg_comp,Wcp_comp] = margin(sys_comp);
disp('Gain margin of compensated system [ratio]')
disp(Gm_comp)
disp('Gain margin of compensated system [dB]')
disp(20*log10(Gm_comp))

disp('Phase margin of compensated system [degrees]')
disp(Pm_comp)

% Check the time domain response of the CL compensated system

sys_comp_cl=tf(OLnum_comp,[0 0 0 OLnum_comp]+OLden_comp);

[y3,t3]=step(sys_comp_cl);


figure,plot(t3,y3,'b')
title('Unit Step Response - Compensated System')
xlabel('Time [s]')
grid
% A rather long tail, Ts about 15 sec.

% Check the ramp response of the CL compensated system
time=0:0.02:50;
y6=lsim(sys_comp_cl,time,time);

figure,plot(time,time,'k',time,y6,'b')
title('Response to Ramp Input - Compensated System')
xlabel('Time [s]')
grid



















