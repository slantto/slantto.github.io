%HW 2 Problem 1
%Sean Lantto

clear all

%choose dataset generated from genGPS function
load('OrbBaseline.mat')

%Setting Constants
N=length(RangeL1); %number of epochs
c=299792458; %speed of light, m/s
nomXYZ=xyzorg;

%excess parameters
trop=0;
ion=0;
mel=0;
llh=0;
Po=0;
To=0;
eo=0;
A1=0;
A2=0;
A3=0;
A4=0;
t=0;
el=0;
klo=0;
tshell=0;
ionfree=0;

%for L1 data
clockBiasNom=0;
for i=1:N
    Satxyz=Sats(1:NumerOfSatsUsed(i),:,i);
    Nsats=NumerOfSatsUsed(i);
    W=eye(NumerOfSatsUsed(i));
    
    %Use function LLSPos to get xyz estimate at each epoch;
    [xyzEst(:,i),clockBiasEst(:,i),PDOP(:,i),TDOP(:,i),GDOP(:,i),prefit,postFit(:,i),TECU(:,i)]=LLSPos(Satxyz,RangeL1(i,1:Nsats),nomXYZ,clockBiasNom,Nsats,c,W,trop,ion,mel,llh,Po,To,eo,A1,A2,A3,A4,t,el,klo,tshell,ionfree);
    
    %substitute xyz estimate for epoch i to be xyz nom for epoch i+1
    nomXYZ=xyzEst(:,i)';
    clockBiasNom=clockBiasEst(:,i)';
    
end
Xyz=Xyz';
load BaselineXYZT.txt;
xyzBL=1000*BaselineXYZT(:,2:4);
xyzBL=xyzBL';
RMSEX=sqrt(mean(((xyzEst(1,:)-xyzBL(1,:)).^2)));
RMSEY=sqrt(mean(((xyzEst(2,:)-xyzBL(2,:)).^2)));
RMSEZ=sqrt(mean(((xyzEst(3,:)-xyzBL(3,:)).^2)));
plot(1:N,xyzEst-xyzBL)
xlabel('epoch');
ylabel('error(m)');
legend('x','y','z');