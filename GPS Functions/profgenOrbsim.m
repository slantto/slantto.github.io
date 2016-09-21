%profgenOrbsim.m
%generates flight profile from STF-1 orbit simulator data
%reads and separates variables
clc
clear

% ampdeg=0 %degrees
% freq=.25 %Hz

% rad2deg=180/pi;
% amp=ampdeg/rad2deg;
load BaselineXYZT.txt;
xyz=BaselineXYZT(:,2:4);
GPST=BaselineXYZT(:,1);
t=
% a=0.01;
% dat(1,:)=[];
% lat = dat(:,3)*(pi/180);
% lon = dat(:,4)*(pi/180);
% alt = dat(:,2);
% utime = dat(:,1);
% pitch = dat(:,5)*(pi/180);
% roll = dat(:,6)*(pi/180);
% yaw = dat(:,7)*(pi/180);

%origin defined as llh position at time 1
% orgllhdeg=[dat(1,3), dat(1,4), dat(1,2)];

%smoothing
% pitch = filter(a, [1 a-1], pitch);
% roll = filter(a, [1 a-1], roll);
% yaw=unwrap(yaw);
% yawf = filter(a, [1 a-1], yaw);
% yawf(1:300)=yaw(1:300)+0.028;
% yaw=yawf;


%converts llh to xyz
% llh=[lat,lon,alt];
enu = zeros(length(xyz),3);

for i=1:length(xyz)
    enu(i,:)=xyz2enu(xyz(i,:),xyz(1,:));
end

%separates x,y,z arrays
e=enu(:,1);
n=enu(:,2);
u=enu(:,3);

%central difference for velocity calculation

ve=gradient(e)./gradient(GPST);
vn=gradient(n)./gradient(GPST);
vu=gradient(u)./gradient(GPST);


%single-timestep velocity calculation for beginning
% ve(1)=(e(2)-e(1))/1;
% vn(1)=(n(2)-n(1))/1;
% vu(1)=(u(2)-u(1))/1;

%single-timestep velocity calculation for end
% ve(length(dat))=e(length(dat))-e(length(dat)-1)/1;
% vn(length(dat))=n(length(dat))-n(length(dat)-1)/1;
% vu(length(dat))=u(length(dat))-u(length(dat)-1)/1;

%zero matrix for 3-column blank space
filler=zeros(length(xyz),3);

%euler angles to dcm
% eul=[roll,pitch,yaw];
% for i=1:(length(eul))
%   b=eulr2dcm([eul(i,:)]);
%   dcm(i,:)=reshape(b',[1,9]);
% end

%put zeroes for dcm for this case
dcm=zeros(length(xyz),3);

%puts arrays together to make profile
prof=horzcat(enu,ve,vn,vu,filler,dcm,utime-utime(1));



% %spline - 1Hz to 200Hz
% t0=(utime(1):0.005:utime(length(utime)))';
% t0=unique(t0);
% 
% utime=unique(utime);
% e(length(dat))=[];
% n(length(dat))=[];
% u(length(dat))=[];
% ve(length(dat))=[];
% vn(length(dat))=[];
% vu(length(dat))=[];
% filler(length(dat))=[];
% roll(length(dat))=[];
% pitch(length(dat))=[];
% yaw(length(dat))=[];
% 
% 
% e2 = interp1(utime,e,t0,'spline');
% n2 = interp1(utime,n,t0,'spline');
% u2 = interp1(utime,u,t0,'spline');
% ve2 = interp1(utime,ve,t0,'spline');
% vn2 = interp1(utime,vn,t0,'spline');
% vu2 = interp1(utime,vu,t0,'spline');
% 
% 
% roll2 = interp1(utime,roll,t0,'spline');
% pitch2 = interp1(utime,pitch,t0,'spline');
% yaw2 = interp1(utime,yaw,t0,'spline');
% 
% %add oscillation
% rollamp=amp; 
% pitchamp=amp;
% yawamp=amp;
% rollfreq=freq;
% pitchfreq=freq;
% yawfreq=freq;
% roll_oscillation = amp*sin(2*pi*t0*rollfreq);
% pitch_oscillation = amp*sin(2*pi*t0*pitchfreq);
% yaw_oscillation =  amp*sin(2*pi*t0*yawfreq);
% 
% roll2 = roll2+roll_oscillation;
% pitch2 = pitch2+pitch_oscillation;
% %yaw2 = yaw2+yaw_oscillation;
% 
% 
% 
% lat(length(lat))=[];
% lon(length(lat))=[];
% alt(length(lat))=[];
% lat2 = interp1(utime,lat,t0,'spline');
% lon2 = interp1(utime,lon,t0,'spline');
% height2 = interp1(utime,alt,t0,'spline');
% 
% 
% %euler angles to dcm
% eul2=[roll2,pitch2,yaw2];
% dcm2=zeros(1449601,9);
% b2=zeros(3,3);
% for i=1:(length(eul2))
%   b2=eulr2dcm([eul2(i,:)]);
%   dcm2(i,:)=reshape(b2',[1,9]);
% end
% filler2=zeros(length(e2),3);
% prof2=horzcat(e2,n2,u2,ve2,vn2,vu2,filler2,dcm2,t0-t0(1));
% orgllhrad=[lat(1), lon(1), orgllhdeg(3)];
% 
% orgllh=[orgllhdeg(1)*pi/180,orgllhdeg(2)*pi/180, orgllhdeg(3)];

Profile=prof;

%ImuData(Profile,102,orgllh);
%Filter_INS_PR_only_ANITA
%gpsReceiver(4000,orgllh,Profile,lat2,lon2,height2,0,-(pi/2),-5,-(pi/2),[0.1,1,0,0.1,1],1,[0,0,0]',1); 



save('GeneratedFlight','Profile','orgllh','lat2','lon2','height2');
