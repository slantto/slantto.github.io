clear all
close all

%problem 7-33
% G(s)=2s+0.1/s(s^2+0.1s+4)

OLnum=[0 0 2 0.1];
OLden=[1 0.1 4 0];
sys_uncomp=tf(OLnum,OLden);

% Design requirements (achieve at least the indicated values):
Kv=4;   % 1/sec
PM=50;   % deg
GM=8;   %dB

% Step #1
% Consider this format for the TF of the Lag Compensator:
% Gc=K*(Ts+1)/(alphaTs+1)

% Step #2
% Determine K from the Kv requirement
% See attached
K=160;
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
disp('Phase margin of G1')
disp(Pm1)
%%
% Step #4
% Determine the location of the gain cross-over frequency,
% which should correspond to the frequency where the phase of G1 is -180 + PMd +(5-12)deg
% Allow 5-12 degrees extra margin to account for extra lag produced by the lag compensator.
phaseG1=-180+PM+12;
[mag,phase,w]=bode(sysG1);
for i=1:length(w);
phase_pr(i)=phase(i);
mag_pr(i)=mag(i);
end
wGC=interp1(phase_pr,w,phaseG1);
disp('Cross-over frequency of the compensated system GcG')
disp(wGC)

%%
% Step #5
% Select the corner frequency coresponding to the zero (1/T) one decade lower
% than the gain cross-over frequency of the compensated system such that the
% phase change produced by the lag compensator is not too large - see Bode diagram
w_zero=wGC/10;
T=1/w_zero;

% Step #6
% Determine the attenuation necessary to bring G1 magnitude down to 0
% at wGC
atten=interp1(w,mag_pr,wGC);
alpha=atten;

% Step #7 Return to old compensator format
Kc=K/alpha;
%%
% Step #8 Verify design
% Check the frequency response of the OL compensated system

OLnum_comp=K*conv([0 0 T 1],OLnum);
OLden_comp=conv([0 0 alpha*T 1],OLden);
sys_comp=tf(OLnum_comp,OLden_comp);

OL_num_lag=Kc*[T 1];
OL_den_lag=[alpha*T 1];
sys_lag=tf(OL_num_lag,OL_den_lag);
figure
bode(sysG1,sys_lag,sys_comp,{0.001,100})
legend('G1','Lag Compensator','Compensated System - OL')
grid


figure
bode(sys_comp,{0.001,100})
title('Bode Diagram of the Compensated System GcG')
grid

[mag_comp,phase_comp,w_comp]=bode(sys_comp,{0.001,100});
[mag_uncomp,phase_uncomp,w_uncomp]=bode(sys_uncomp,w_comp);
[mag_G1,phase_G1,w_G1]=bode(sysG1,w_comp);

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

figure,margin(sys_comp)

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

figure,plot(t1,y1,'k',t3,y3,'r')
title('Unit Step Response')
legend('Uncompensated System','Compensated System')
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



















