%HW 3 Problem 1
%Sean Lantto

clear all

load('dataSet3.mat')

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

ion=input('Include Ionosphere delay(1 for yes, 0 for no)');
if ion==1
    A1=input('define A1(sec)');
    A2=input('define A2(sec)');
    A3=input('define A3(sec)');
    A4=input('define A4(sec)');
    t=
end

%For P1 data
for i=1:N
    %Seperate out pseudorange and Satxyz for epoch i
    pr=prDataP1(1:nSat(i),i);
    Satxyz=satsXYZ(1:nSat(i),:,i);
    Nsats=nSat(i);
    W=zeros(nSat(i));
    llh=xyz2llh(xyz);
    %Create weight matrix
    for x=1:nSat(i)
        Satenu(x,:)=xyz2enu(Satxyz(x,:),xyznom);
        sinel(x,:)=(Satenu(x,3))/(norm(Satenu(x,:)));
        el(x,i)=asind(sinel(x,:)); %elevation angle
        W(x,x)=sinel(x,:); %weighting matrix as a function of elevation angle
        mel(x)=1/(sqrt(1-((cosd(el(x,i))/1.001).^2))); %mapping function for troposphere delay as a function of elevation angle
    end
    
    
    %Use function LLSPos to get xyz estimate at each epoch
    [xyzEstP1(:,i),clockBiasEstP1(:,i),PDOP(:,i),TDOP(:,i),GDOP(:,i),preFit,postFit]=LLSPos(Satxyz,pr,nomXYZ,clockBiasNom,Nsats,c,W,trop,ion,mel,llh,Po,To,eo);
    %substitute xyz estimate for epoch i to be xyz nom for epoch i+1
    nomXYZ=xyzEstP1(:,i)';
    clockBiasNom=clockBiasEstP1(:,i)';
    
    
    %Find Truth ENU
    xyz=truthXYZ(:,i);
    orgxyz=xyznom;
    enu(:,i)=xyz2enu(xyz,orgxyz);
    truthenu(:,i)=enu(:,i);
    
    %find ENU 
    xyz=xyzEstP1(:,i);
    orgxyz=xyznom;
    enu(:,i)=xyz2enu(xyz,orgxyz);
    estenuP1(:,i)=enu(:,i);
    
   
    
end

%plot enu error
figure
plot(1:N,(estenuP1-truthenu))

xyz=nomXYZ;
%For P2 data
for i=1:N
    %Seperate out pseudorange and Satxyz for epoch i
    pr=prDataP2(1:nSat(i),i);
    Satxyz=satsXYZ(1:nSat(i),:,i);
    Nsats=nSat(i);
    W=zeros(nSat(i));
    llh=xyz2llh(xyz);
    %Create weight matrix
    for x=1:nSat(i)
        Satenu(x,:)=xyz2enu(Satxyz(x,:),xyznom);
        sinel(x,:)=(Satenu(x,3))/(norm(Satenu(x,:)));
        el(x,i)=asind(sinel(x,:));
        W(x,x)=sinel(x,:);
    end
    %Use function LLSPos to get xyz estimate at each epoch
    [xyzEstP2(:,i),clockBiasEstP2(:,i),PDOP(:,i),TDOP(:,i),GDOP(:,i)]=LLSPos(Satxyz,pr,nomXYZ,clockBiasNom,Nsats,c,W,trop,ion,mel,llh,Po,To,eo);
    %substitute xyz estimate for epoch i to be xyz nom for epoch i+1
    nomXYZ=xyzEstP2(:,i)';
    clockBiasNom=clockBiasEstP2(:,i)';
    
    
    xyz=xyzEstP2(:,i);
    orgxyz=xyznom;
    enu(:,i)=xyz2enu(xyz,orgxyz);
    estenuP2(:,i)=enu(:,i);
    
   
    
end

figure
plot(1:N,(estenuP2-truthenu))


figure
plot(1:N,(estenuP1-estenuP2))
