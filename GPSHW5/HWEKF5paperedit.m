%EKF for Differential GPS dats
%Ver. 5.0
clc
clear
close all

load realData.mat

%% Sort Orbit Data into a usable format
format long
a = load ('orbitData.txt');
c = zeros(32,5,12000);

for k = 2:12001
    Satxyz(:,:,k-1) = sortrows(a(1+(32*(k-2)):32*(k-1),:),1);
end
Satxyz(:,3:5,:)=Satxyz(:,3:5,:)*1000;


%Base station position from PPP
tru_usrxyzO=[856293.742375760 -4843032.97989093 4048019.36292819];

%% find matching timetags, base station is 1Hz, rover is 10Hz

%Finds all matching times between rover and base station
C=intersect(base.TOWSEC, rover.TOWSEC);
k=0;
%creates arrays with the indicies of the matching times in both base and rover
%of the matching time tages, this becomes useful later
for jj=1:length(C)
    if nnz(rover.C1(:,find(rover.TOWSEC == C(jj))))<5
    else
        k=k+1;
        
        [rows,cols(:,k)] = find(Satxyz==C(jj));
        
        CMB(k,:)=find(base.TOWSEC == C(jj));
        CMR(k,:)=find(rover.TOWSEC == C(jj));
        CMS(k,:)=find(Satxyz(1,2,:) == C(jj));
        %create nSat array and an array of Sats available at each epoch
        nSat(:,k)=nnz(rover.C1(:,CMR(k)));
        SVID(1:nSat(k),k) = find(rover.C1(:,CMR(k)));
        
    end
end

%Initial number of states
numNonAmbStates=3; %Position just position
numAmbStates=2*(nSat(:,1)-1); %ambiguity states for 1st epoch
numStates(1,:) = numAmbStates + numNonAmbStates;

numEpochs=length(CMB);

%Estimate error covariance matrix
P= diag([1,1,1,ones(1,numAmbStates)*100^2]);

% Measurment error covariance matrix
R = (2.5^2)*(ones(numAmbStates,numAmbStates)+eye(numAmbStates));

% state transition matrix is just identity
F= eye(numNonAmbStates+ numAmbStates);

%Process error covarince matrix
%platform is moving up to ~1 m/s 
Q=blkdiag(eye(numNonAmbStates)*1^2, eye(numAmbStates,numAmbStates)) ;

%% Set some important constancts
freq_L1= 1575.42e6;
freq_L2 = 1227.6e6;

speedOfLight = 299792458;
lambdaL1=speedOfLight/1575.42e6;
lambdaL2=speedOfLight/1227.6e6;

%% Changing zeros to NaNs to prevent differencing errors
notFixed=0; % counter that tells ups how many epochs were fixed succesfully by LAMBDA
rover.C1(rover.C1==0)=NaN;
rover.P2(rover.P2==0)=NaN;
rover.PHASE1(rover.PHASE1==0)=NaN;
rover.PHASE2(rover.PHASE2==0)=NaN;

base.C1(base.C1==0)=NaN;
base.P2(base.P2==0)=NaN;
base.PHASE1(base.PHASE1==0)=NaN;
base.PHASE2(base.PHASE2==0)=NaN;

%% single differencing
for j=1:length(CMR)
 
    SDRangeL1(:,j)=rover.C1(1:30,CMR(j))-base.C1(:,CMB(j));
    SDRangeL2(:,j)=rover.P2(1:30,CMR(j))-base.P2(:,CMB(j));
    SDPhaseL1(:,j)=rover.PHASE1(1:30,CMR(j))-base.PHASE1(:,CMB(j));
    SDPhaseL2(:,j)=rover.PHASE2(1:30,CMR(j))-base.PHASE2(:,CMB(j));
    
end
%creates an array of reference satellites
[refsat,satsXYZ] = referenceSAT(SVID,Satxyz,tru_usrxyzO,cols);

%% double differencing

for  j=1:length(CMR)
k=0;
    for i=1:30
        if i~=refsat(j)
            k=k+1;

            DDRangeL1(k,j)=SDRangeL1(i,j)-SDRangeL1(refsat(j),j);
            DDRangeL2(k,j)=SDRangeL2(i,j)-SDRangeL2(refsat(j),j);
            DDPhaseL1(k,j)=SDPhaseL1(i,j)-SDPhaseL1(refsat(j),j);
            DDPhaseL2(k,j)=SDPhaseL2(i,j)-SDPhaseL2(refsat(j),j);
            SVIDdd(k,j)=i*(isfinite(DDRangeL1(k,j)));
        end
    end
end

%% Enter Kalman Filter
SVdd(1,:)=nonzeros(SVIDdd(:,1));
x_float=zeros((numNonAmbStates+(2*max(nSat))),length(CMR));
x_float(1:numStates,1)=[0,0,0,randi([-10,10])*ones(1,numAmbStates)]';
Fixed=zeros(2*max(nSat),length(CMR));
x_fixed=zeros((numNonAmbStates+(2*max(nSat))),length(CMR));
for i= 2:length(CMR)
    %New number of states dependent on number of Sats
    numAmbStates=2*(numel(nonzeros(SVIDdd(:,i))));
    numStates(i,:)=numAmbStates+numNonAmbStates;
    % Prediction Step
    % Reformulate Covariance matrices if Sats are added or dropped
    [x_float(1:numStates(i,:),i),P]=addRemoveBiasParm(x_float(1:numStates(i-1,:),i-1),P,numNonAmbStates,nonzeros(SVIDdd(:,i-1)),nonzeros(SVIDdd(:,i)),100^2);
    F=eye(numStates(i,:));
    Q=blkdiag(eye(numNonAmbStates)*1^2, eye(numAmbStates,numAmbStates)) ;
    
    P = F*P*F' + Q;
    
    % measurement update
    % allocate size of H
    H=zeros(numAmbStates, numStates(i,:));
    
    % fill in the difference between unit vector part in the first 3
    % columns of H
    
    % calculate the unit vector to the ref sat, this now changes each epoch
    RefSat=satsXYZ(refsat(i),:,i);
    vec2RefSat=((RefSat-tru_usrxyzO)/norm(RefSat-tru_usrxyzO));
    
    % measurement vector only carrier phase double differences
    % note that numDD = nSat-1;
    z=[DDPhaseL1(find(isfinite(DDPhaseL1(:,i))==1),i)'*lambdaL1 DDPhaseL2(find(isfinite(DDPhaseL2(:,i))==1),i)'*lambdaL2]';
    if length(z)~=numAmbStates
        z(length(z):numAmbStates,:)=0;
    end
    
    for j=1:numAmbStates;
        
        
        R = (2.5^2)*(ones(numAmbStates,numAmbStates)+eye(numAmbStates));

        
        SVdd=nonzeros(SVIDdd(:,i))';
        
        %  indexing stragety to accomodate order of measurments
        % is all L1 then all L2, so must repeat PRNs
        Ind=mod(j-1,numel(nonzeros(SVIDdd(:,i))))+1;
        Ssat=[Satxyz(SVdd(:,Ind),3,i) Satxyz(SVdd(:,Ind),4,i) Satxyz(SVdd(:,Ind),5,i)];
        vec2DDSat=((Ssat-tru_usrxyzO)/norm(Ssat-tru_usrxyzO));
        
        H(j,1:3)=-(vec2DDSat-vec2RefSat);
        %half of ambiguity states are for L1 and the other L2
        if j<=numAmbStates/2
            H(j,3+j) = lambdaL1;
        else
            H(j,3+j) = lambdaL2;
        end
    end
    
    % calculate Kalman Gain
    K = P*H'*inv(H*P*H'+R);
    
    % update state
    x_float(1:numStates(i,:),i) =  x_float(1:numStates(i,:),i) + K*(z-H*x_float(1:numStates(i,:),i));
    
    % update covariance
    P=(eye(numNonAmbStates+ numAmbStates) - K*H)*P;
    
    
     % apply the LAMBDA method in parallel
    [fix,sqr,Ps,Qz,Z,nFix,mu]=LAMBDA(x_float(numNonAmbStates+1:numStates(i,:),i), ... 
            P(numNonAmbStates+1:numStates(i,:),numNonAmbStates+1:numStates(i,:)),6,'ncands',2);
    
     Fixed(1:numAmbStates,i)=fix(:,1);
     if(nFix>0)
        x_fixed(1:numNonAmbStates,i)=x_float(1:numNonAmbStates,i)- ... 
            P(1:numNonAmbStates,numNonAmbStates+1:numStates(i,:))* ... 
            pinv(P(numNonAmbStates+1:numStates(i,:),numNonAmbStates+1:numStates(i,:)))* ... 
            (x_float(numNonAmbStates+1:numStates(i,:),i)-fix(:,1));
        
        x_fixed(numNonAmbStates+1:numStates(i,:),i)=Fixed(1:numAmbStates,i);
     else
        % LAMBDA method was unable to correctly fix integer ambiguities
        notFixed=notFixed+1;
        x_fixed(1:numNonAmbStates,i)=x_float(1:numNonAmbStates,i);
        x_fixed(numNonAmbStates+1:numStates(i,:),i)=Fixed(1:numAmbStates,i);
    end
    
    
end


% compute absolute position estimate, float and fixed and plot comparisons
for i=1:length(CMR)
    
    xyzEst(:,i)=tru_usrxyzO'+x_float(1:3,i);
    estENU(:,i)=xyz2enu(xyzEst(:,i),tru_usrxyzO);
    
    xyzFix(:,i)=tru_usrxyzO'+x_fixed(1:3,i);
    fixENU(:,i)=xyz2enu(xyzFix(:,i),tru_usrxyzO);
    
end

time=base.TOWSEC(CMB);
ENUFloat=horzcat(time',estENU');
ENUFIXED=horzcat(time',fixENU');
XYZFLOAT=horzcat(time',xyzEst');
XYZFIX=horzcat(time',xyzFix');

[rtkENU,rtkXYZ]=rtklibENU(tru_usrxyzO);
[GYPSYenu, GYPSYxyz]=GYPSYenu(tru_usrxyzO);

%difference between EKF,RTK, and GYPSY results
inter=intersect(time,rtkENU(:,1));
[rover,text] = xlsread('roverRTK.xlsx');
k=1;
for i=1:length(inter)
                
        roverROW=find(rover(:,2) == inter(i));
        XYZFLOATrow=find(XYZFLOAT == inter(i));
        XYZFIXrow=find(XYZFIX == inter(i));
        XYZ_float_diff(k,:) = rover(roverROW,3:5)-XYZFLOAT(XYZFLOATrow,2:4);
        XYZ_fix_diff(k,:) = rover(roverROW,3:5)-XYZFIX(XYZFIXrow,2:4);
   k=k+1;
end

figure
plot(inter,XYZ_float_diff(:,1),'k',inter,XYZ_float_diff(:,3),'r',inter,XYZ_float_diff(:,2),'b')
title('Difference between EKF Float solution and RTKlib Solution')
legend('X dir','Z dir','Y dir')
xlabel('TOW Second');
ylabel('Difference in Meters');

figure
plot(inter,XYZ_fix_diff(:,1),'k',inter,XYZ_fix_diff(:,3),'r',inter,XYZ_fix_diff(:,2),'b')
title('Difference between EKF Fixed solution and RTKlib Solution')
legend('X dir','Z dir','Y dir')
xlabel('TOW Second');
ylabel('Difference in Meters');

figure
plot3(rtkXYZ(:,1),rtkXYZ(:,2),rtkXYZ(:,3))
hold on
plot3(xyzFix(1,:),xyzFix(2,:),xyzFix(3,:))
hold off
xlabel('X');ylabel('Y');zlabel('Z');
legend('RTK Position','EKF Position');
title('Comparison of RTK position and EKF position in ECEF')

figure
plot3(fixENU(1,:),fixENU(2,:),fixENU(3,:))
title('EKF ENU Solution');


figure
plot3(estENU(1,:),estENU(2,:),estENU(3,:));
view(2);


