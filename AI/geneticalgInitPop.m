Rearth=6371; %km radius of earth
mu=3.986004415*10^5; %km^3/s^2 Gravitational parameter of earth
R0=5000+Rearth;%Initial orbit at 5000km altitude
Rf=20000+Rearth; %Desired orbit at 20000km altitude
lower=500+Rearth;%lower limit for transfer orbit radii
upper=50000+Rearth;%upper limit for transfer orbit radii


numInd=100;

dV1=zeros(1,numInd);
dV2=dV1;
dV3=dV1;
dV4=dV1;
R2=dV1;
R3=dV1;
TOF=zeros(1,numInd);
for k=1:numInd
numVar=2;
Trans=lower+(upper-lower).*rand(1,numVar);
R2(k)=Trans(:,1);
R3(k)=Trans(:,2);

x=Rf/R0;
y=R2(k)/R0;
z=R3(k)/R0;

dV1(k)=abs(sqrt(mu/R0)*(sqrt(((2*y)/(1+y)))-1));
dV2(k)=abs(sqrt(mu/R0)*sqrt(1/y)*(sqrt((2*z)/(z+y))-sqrt(2/(1+y))));
dV3(k)=abs(sqrt(mu/R0)*sqrt(1/z)*(sqrt((2*y)/(z+y))-sqrt((2*x)/(z+x))));
dV4(k)=abs(sqrt(mu/R0)*sqrt(1/x)*(sqrt(((2*x)/(z+x)))-1));
TOF1=pi*sqrt(((R0+R2(k))^3)/(8*mu));
TOF2=pi*sqrt(((R2(k)+R3(k))^3)/(8*mu));
TOF3=pi*sqrt(((R3(k)+Rf)^3)/(8*mu));
TOF(k)=TOF1+TOF2+TOF3;% time of flight in seconds
end
R2=R2';
R3=R3';
TOF=TOF'/3600;%puts time of flight in hours
dV=[dV1',dV2',dV3',dV4'];%all delta-v's are in km/s
dVtotes=dV1'+dV2'+dV3'+dV4';

Chromo=horzcat(dV,TOF);


