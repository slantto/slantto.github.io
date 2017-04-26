%HW 2 Problem 1
%Sean Lantto

clear all

load('dataSet4.mat')

%Setting Constants
N=2400; %number of epochs
c=299792458; %speed of light, m/s
xyznom=nomXYZ;
xyz=xyznom;
%Atmospheric considerations
trop=input('Include Troposphere delay(1 for yes 0 for no)');
if trop==1
    Po=input('what is your total pressure?(mbar)');
    To=input('what is your temperature?(kelvin)');
    eo=input('what is your partial pressure do to water?(mbar)');
else
    Po=0;
    To=0;
    eo=0;
end
A1=0;
A2=0;
A3=0;
A4=0;
klo=0;
tshell=0;

ion=input('Include Ionosphere delay(1 for yes, 0 for no)');
if ion==1
    
    ionfree=0;
    
    A1=input('define A1(sec)');
    A2=input('define A2(sec)');
    A3=input('define A3(sec)');
    A4=input('define A4(sec)');
    timeoffset=input('How many hours since begining of gps week?');
    timeoffset=timeoffset*3600; %convert to seconds
    
    whichion=input('1 for klorbuchar, 2 for thin shell');
    if whichion==1
        klo=1;
        
    elseif whichion==2
        tshell=1;
        
    end
elseif ion==0
    TECU=0;
    ionfree=input('Would you like to use the dual-frequency iono-free model?(1=yes, 0=no)');
end


%For P1 data
for i=1:N
    %Seperate out pseudorange and Satxyz for epoch i
    if ionfree==0
        pr=prDataP1(1:nSat(i),i);
    elseif ionfree==1
        pr=(2.546*prDataP1(1:nSat(i),i))-(1.546*prDataP2(1:nSat(i),i));
    end
    Satxyz=satsXYZ(1:nSat(i),:,i);
    Nsats=nSat(i);
    W=zeros(nSat(i));
    llh=xyz2llh(xyz);
    if ion==1
        t=mod(time(i),86400)+timeoffset;
    else
        t=0;
    end
    %Create weight matrix
    for x=1:nSat(i)
        Satenu(x,:)=xyz2enu(Satxyz(x,:),xyznom);
        sinel(x,:)=(Satenu(x,3))/(norm(Satenu(x,:)));
        el(x,i)=asind(sinel(x,:)); %elevation angle
        W(x,x)=sinel(x,:); %weighting matrix as a function of elevation angle
        mel(x)=1/(sqrt(1-((cosd(el(x,i))/1.001).^2))); %mapping function for troposphere delay as a function of elevation angle
    end
    
    %Use function LLSPos to get xyz estimate at each epoch;

    [xyzEst(:,i),clockBiasEst(:,i),PDOP(:,i),TDOP(:,i),GDOP(:,i),prefit,postFit(:,i),TECU(:,i)]=LLSPos(Satxyz,pr,nomXYZ,clockBiasNom,Nsats,c,W,trop,ion,mel,llh,Po,To,eo,A1,A2,A3,A4,t,el(:,i),klo,tshell,ionfree);

    %substitute xyz estimate for epoch i to be xyz nom for epoch i+1
    nomXYZ=xyzEst(:,i)';
    clockBiasNom=clockBiasEst(:,i)';
    for x=1:nSat(i)
        preFit(x,i)=prefit(x);
    end
    
    %Find Truth ENU
    xyz=truthXYZ(:,i);
    orgxyz=xyznom;
    enu(:,i)=xyz2enu(xyz,orgxyz);
    truthenu(:,i)=enu(:,i);
    
    %find ENU
    xyz=xyzEst(:,i);
    orgxyz=xyznom;
    enu(:,i)=xyz2enu(xyz,orgxyz);
    estenu(:,i)=enu(:,i);
<<<<<<< HEAD
    
    
=======
    
    RMSEenu(1,i)=sqrt(mean((estenu(1,i)-truthenu(1,i)).^2));
    RMSEenu(2,i)=sqrt(mean((estenu(2,i)-truthenu(2,i)).^2));
    RMSEenu(3,i)=sqrt(mean((estenu(3,i)-truthenu(3,i)).^2));
    RMSEclkB(i)=sqrt(mean((clockBiasEst(i)-(truthClockBias(i)/c)).^2));
    RMSE3d(:,i)=sqrt(mean((estenu(:,i)-truthenu(:,i)).^2));
>>>>>>> 49326cf997cebea9f2b27102cea3d6b0bc877ad6
    
end

%plot enu error
figure
diffenu=(estenu-truthenu);
diffclkB=clockBiasEst-(truthClockBias/c);
subplot(211)
plot(1:N,diffenu)
ylabel('error(m)')
xlabel('epoch')
legend('E','N','U')
title('Estimated ENU error(DataSet4)')
subplot(212)
plot(1:N,diffclkB)
ylabel('error(s)')
xlabel('epoch')
title('Estimated clock bias error(DataSet4)')

<<<<<<< HEAD
RMSEenu(1)=sqrt(mean((estenu(1,:)-truthenu(1,:)).^2));
RMSEenu(2)=sqrt(mean((estenu(2,:)-truthenu(2,:)).^2));
RMSEenu(3)=sqrt(mean((estenu(3,:)-truthenu(3,:)).^2));
RMSEclkB=sqrt(mean((clockBiasEst-(truthClockBias/c)).^2));
RMSE3d=sqrt(mean((estenu-truthenu).^2));
=======
figure
subplot(311)
plot(1:N,RMSEenu)
ylabel('error(m)')
xlabel('epoch')
legend('E','N','U')
title('RMSE ENU(DataSet4)')
subplot(312)
plot(1:N,RMSE3d)
ylabel('error(m)')
xlabel('epoch')
title('RMSE 3D(DataSet4)')
subplot(313)
plot(1:N,RMSEclkB)
ylabel('error(s)')
xlabel('epoch')
title('RMSE Clock Bias(DataSet4)')
>>>>>>> 49326cf997cebea9f2b27102cea3d6b0bc877ad6

figure
plot(1:N,(prDataP1-prDataP2))
ylabel('error(m)')
xlabel('epoch')
title('Difference in observed pseudorange on L1 and L2(DataSet4)')