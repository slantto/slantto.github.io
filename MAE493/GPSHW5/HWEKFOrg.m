clc
clear
load realData.mat

format long
a = load ('orbitData.txt');
c = zeros(32,5,12000);

for k = 2:12001
    Satxyz(:,:,k-1) = sortrows(a(1+(32*(k-2)):32*(k-1),:),1);
end
display(c(:,:,12000));

tru_usrxyzO=[856293.742375760 -48430329.7989093 4048019.36292819];

%find matching timetags, base station is 1Hz, rover is 10Hz
C=intersect(base.TOWSEC, rover.TOWSEC);
k=0;
for jj=1:length(C)
    if nnz(rover.C1(:,find(rover.TOWSEC == C(jj))))<5
    else
        k=k+1;
        
        CMB(k,:)=find(base.TOWSEC == C(jj));
        CMR(k,:)=find(rover.TOWSEC == C(jj));
        CMS(k,:)=find(Satxyz(1,2,:) == C(jj));
        
        nSat(:,k)=nnz(rover.C1(:,CMR(k)));
        SVID(1:nSat(k),k) = find(rover.C1(:,CMR(k)));
       Sats(:,:,k)=Satxyz(:,:,CMS(k,:)); 
       
    end
end
Satxyz=Sats;

numNonAmbStates=3; %Position 
numAmbStates=2*(nSat(:,1)-1); %ambiguity states for 1st epoch
numStates(1,:) = numAmbStates + numNonAmbStates;


numEpochs=length(CMB);


%
P= diag([1,1,1,ones(1,numAmbStates)*1000^2]);


% 
R = (.02^2)*(ones(numAmbStates,numAmbStates)+eye(numAmbStates));

% state transition matrix is just PRNentity
F= eye(numNonAmbStates+ numAmbStates);

% platform is moving up to 2 m/s and ambigutities are constant
Q=blkdiag(eye(numNonAmbStates)*2^2, zeros(numAmbStates,numAmbStates)) ;

% initialize amb estimates with DD pseudorange measurements
freq_L1= 1575.42e6;
freq_L2 = 1227.6e6;

speedOfLight = 299792458;
lambdaL1=speedOfLight/1575.42e6;
lambdaL2=speedOfLight/1227.6e6;

notFixed=0; % counter that tells ups how many epochs were fixed succesfully by LAMBDA
rover.C1(rover.C1==0)=NaN;
rover.P2(rover.P2==0)=NaN;
rover.PHASE1(rover.PHASE1==0)=NaN;
rover.PHASE2(rover.PHASE2==0)=NaN;

base.C1(base.C1==0)=NaN;
base.P2(base.P2==0)=NaN;
base.PHASE1(base.PHASE1==0)=NaN;
base.PHASE2(base.PHASE2==0)=NaN;

refsat=21;
%single differencing
for j=1:length(CMR)
 
    SDRangeL1(:,j)=rover.C1(1:30,CMR(j))-base.C1(:,CMB(j));
    SDRangeL2(:,j)=rover.P2(1:30,CMR(j))-base.P2(:,CMB(j));
    SDPhaseL1(:,j)=rover.PHASE1(1:30,CMR(j))-base.PHASE1(:,CMB(j));
    SDPhaseL2(:,j)=rover.PHASE2(1:30,CMR(j))-base.PHASE2(:,CMB(j));
    
end



%double differencing with refsat=21

k=0;
for i=1:30
    if i~=refsat
        k=k+1;
  
        DDRangeL1(k,:)=SDRangeL1(i,:)-SDRangeL1(refsat,:);
        DDRangeL2(k,:)=SDRangeL2(i,:)-SDRangeL2(refsat,:);
        DDPhaseL1(k,:)=SDPhaseL1(i,:)-SDPhaseL1(refsat,:);
        DDPhaseL2(k,:)=SDPhaseL2(i,:)-SDPhaseL2(refsat,:);
        SVIDdd(k,1:length(CMR))=i*(isfinite(DDRangeL1(k,:)));
    
    end
    
end

DDRangeL1=horzcat(DDRangeL1(:,1:734),DDRangeL1(:,738:848));
DDRangeL2=horzcat(DDRangeL2(:,1:734),DDRangeL2(:,738:848));
DDPhaseL1=horzcat(DDPhaseL1(:,1:734),DDPhaseL1(:,738:848));
DDPhaseL2=horzcat(DDPhaseL2(:,1:734),DDPhaseL2(:,738:848));
SVIDdd=horzcat(SVIDdd(:,1:734),SVIDdd(:,738:848));

SVdd(1,:)=nonzeros(SVIDdd(:,1));
x_float=zeros((numNonAmbStates+(2*max(nSat))),length(CMR)-4);
x_float(1:numStates,1)=[0,0,0,randi([-10,10])*ones(1,numAmbStates)]';
Fixed=zeros(2*max(nSat),length(CMR)-4);
x_fixed=zeros((numNonAmbStates+(2*max(nSat))),length(CMR)-4);
for i= 2:length(CMR)-4
    numAmbStates=2*(numel(nonzeros(SVIDdd(:,i))));
    numStates(i,:)=numAmbStates+numNonAmbStates;
    % prediction
    [x_float(1:numStates(i,:),i),P]=addRemoveBiasParm(x_float(1:numStates(i-1,:),i-1),P,numNonAmbStates,nonzeros(SVIDdd(:,i-1)),nonzeros(SVIDdd(:,i)),1000^2);
    F=eye(numStates(i,:));
    Q=blkdiag(eye(numNonAmbStates)*100^2, zeros(numAmbStates,numAmbStates)) ;
    
    %x_float(1:numStates(i,:),i)=F*x_float(1:numStates(i,:),i-1);
    P = F*P*F' + Q;
    
    % measurement update
    % allocate size of H
    H=zeros(numAmbStates, numStates(i,:));
    
    % fill in the difference between unit vector part in the first 3
    % columns of H
    
    % calculate the unit vector to the ref sat, this only needed once
    RefSat=[Satxyz(refsat,3,i) Satxyz(refsat,4,i) Satxyz(refsat,5,i)];
    vec2RefSat=((RefSat-tru_usrxyzO)/norm(RefSat-tru_usrxyzO));
    
    % measurement vector only carrier phase double differences
    % note that numDD = nSat-1;
    z=[DDPhaseL1(find(isfinite(DDPhaseL1(:,i))==1),i)' DDPhaseL2(find(isfinite(DDPhaseL2(:,i))==1),i)']';
    if length(z)~=numAmbStates
        z(length(z):numAmbStates,:)=0;
    end
    
    for j=1:numAmbStates;
        
        
        R = (2^2)*(ones(numAmbStates,numAmbStates)+eye(numAmbStates));

        
        SVdd=nonzeros(SVIDdd(:,i))';
        
        %  indexing stragety to accomodate order of measurments
        % is all L1 then all L2, so must repeat PRNs
        Ind=mod(j-1,numel(nonzeros(SVIDdd(:,i))))+1;
        Ssat=[Satxyz(SVdd(:,Ind),3,i) Satxyz(SVdd(:,Ind),4,i) Satxyz(SVdd(:,Ind),5,i)];
        vec2DDSat=((Ssat-tru_usrxyzO)/norm(Ssat-tru_usrxyzO));
        
        H(j,1:3)=-(vec2DDSat-vec2RefSat);
        
        % columns 4:end and row 1:nObs is an PRNentity matrix times the
        % wavelength of the double difference observation ( need to go from units of cycles to
        % units of meters )
        % if first half ob measurements, we are processing L1 data, L2
        % for 2nd half
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
for i=1:length(CMR)-4
    
    xyzEst(:,i)=tru_usrxyzO'+x_float(1:3,i);
    estENU(:,i)=xyz2enu(xyzEst(:,i),tru_usrxyzO);
    
    xyzFix(:,i)=tru_usrxyzO'+x_fixed(1:3,i);
    fixENU(:,i)=xyz2enu(xyzFix(:,i),tru_usrxyzO);
    
end

figure,
plot3(estENU(1,:),estENU(2,:),estENU(3,:))
figure
plot3(xyzEst(1,:),xyzEst(2,:),xyzEst(3,:))
view(2);
% hold on
% plot(fixENU)
% hold off
% plot(fixENU(1,:),fixENU(2,:),'g','lineWPRNth',3)
% xlabel('East (m)')
% ylabel('North (m)')
% legend('Truth','Float','Fixed')
% grPRN on
% grPRN on
% 
% 
% figure,subplot(311),plot((estENU(1,:)-trueENU(1,:))')
% hold on
% grPRN on
% plot((fixENU(1,:)-trueENU(1,:))',':','lineWPRNth',3)
% legend('Float Solution', 'Fixed Solution')
% ylabel('East (m)')
% 
% subplot(312),plot((estENU(2,:)-trueENU(2,:))')
% hold on
% grPRN on
% plot((fixENU(2,:)-trueENU(2,:))',':','lineWPRNth',3)
% legend('Float Solution', 'Fixed Solution')
% ylabel('North (m)')
% 
% subplot(313),plot((estENU(3,:)-trueENU(3,:))')
% hold on
% grPRN on
% plot((fixENU(3,:)-trueENU(3,:))',':','lineWPRNth',3)
% legend('Float Solution', 'Fixed Solution')
% ylabel('Up (m)')
% 

