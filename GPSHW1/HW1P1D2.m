%HW 1, Part 1, Data Set 2
%Sean Lantto


% %estimate position with iteration of the nominal position

clear all
N=2400; %number of epochs
c=299792458; %speed of light, m/s

%estimate position and clock bias for data set 2
load('dataSet2.mat')
xyznom=nomXYZ;

for i=1:N
    %Seperate out pseudorange and Satxyz for epoch i
    pr=prData(1:nSat(i),i);
    Satxyz=satsXYZ(1:nSat(i),:,i);
    Nsats=nSat(i);
    W=zeros(nSat(i));
    %Create weight matrix
    for x=1:nSat(i)
        Satenu(x,:)=xyz2enu(Satxyz(x,:),xyznom);
        sinel(x,:)=(Satenu(x,3))/(norm(Satenu(x,:)));
        el(x,i)=asind(sinel(x,:));
        W(x,x)=sinel(x,:);
    end
    %Use function LLSPos to get xyz estimate at each epoch
    [xyzEst(:,i),clockBiasEst(:,i),PDOP(:,i),TDOP(:,i),GDOP(:,i)]=LLSPos(Satxyz,pr,nomXYZ,clockBiasNom,Nsats,c,W);
    %substitute xyz estimate for epoch i to be xyz nom for epoch i+1
    nomXYZ=xyzEst(:,i)';
    clockBiasNom=clockBiasEst(:,i)';
    
    xyz=truthXYZ(:,i);
    orgxyz=xyznom;
    enu(:,i)=xyz2enu(xyz,orgxyz);
    truthenu(:,i)=enu(:,i);
    
    xyz=xyzEst(:,i);
    orgxyz=xyznom;
    enu(:,i)=xyz2enu(xyz,orgxyz);
    estenu(:,i)=enu(:,i);
    
end



PosErrxyz=xyzEst-truthXYZ;

figure
plot(1:N,PosErrxyz)
xlabel('Data epoch')
ylabel('error(m)')
legend('X','Y','Z')
title('Position Error for dataSet2 w/ nom iteration')

PosErrenu=estenu-truthenu;

figure
plot(1:N,PosErrenu)
xlabel('Data epoch')
ylabel('error(m)')
legend('E','N','U')
title('Position Error for dataSet2 w/ nom iteration ')

for i=1:N
    PosErrxyzmag(:,i)=norm(PosErrxyz(i));
end
theoerr=sigma_URE*PDOP;

figure
plot(1:N,PosErrxyzmag,1:N,theoerr)
xlabel('Data epoch')
ylabel('error(m)')
legend('actual error','theoretical error')
title('Comparison of PDOP/URE and actual error(dataset2)')

ClockErr=clockBiasEst-(truthClockBias/c);
theoclkerr=TDOP*sigma_URE;
figure
plot(1:N,ClockErr,1:N,theoclkerr)
xlabel('Data epoch')
ylabel('error(sec)')
legend('actual error','theoretical error')
title('Comparison of TDOP/URE and actual error(dataset2)')