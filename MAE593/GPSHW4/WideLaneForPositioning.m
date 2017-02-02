clear
load dgpsData
close all

%% Example of DGPS positioning with Dual Frequency Wide Lanining
% Author: Jason N. Gross

%% Note that this is simplified because
% Both user and reference station track same sats and are stored in
% same configuration


% constants
lambda_L1=.19;
lambda_L2=0.244;
lambda_L12=.862;
alpha=lambda_L2/lambda_L1;

%% add a cycle slip
phaseL1_U(3,500:end)=phaseL1_U(3,500:end)+50*lambda_L1;

%% Step #1
% form Wide Lane Phase data
%% I gave you phase data in meters, convert back to cycles

WL_phase_U=phaseL1_U/lambda_L1-phaseL2_U/lambda_L2;
WL_phase_R=phaseL1_R/lambda_L1-phaseL2_R/lambda_L2;

WL_phase_U=phaseL1_U/lambda_L1-phaseL2_U/lambda_L2;
WL_phase_R=phaseL1_R/lambda_L1-phaseL2_R/lambda_L2;
% Step #2
%form single difference
WL_SD_phase=WL_phase_U-WL_phase_R;

%% Step #3
% form L1 range SD and L1 and L2 DD ( needed later )
L1_SD_range=prDataP1_U-prDataP1_R;
L2_SD_range=prDataP2_U-prDataP2_R;

L1_SD_phase=phaseL1_U-phaseL1_R;
L2_SD_phase=phaseL2_U-phaseL2_R;
%%% Step #4
% choose a high-elevation sat as reference satellite
% (I chose #4 PRN15)
% form double differences

% Note there are 8 sats
refInd=6;
k=0;
for i=1:8
    if i~=refInd
        k=k+1;
        WL_DD_Phase(k,:)=WL_SD_phase(i,:)-WL_SD_phase(refInd,:);
        L1_DD_Range(k,:)=L1_SD_range(i,:)-L1_SD_range(refInd,:);
        L2_DD_Range(k,:)=L2_SD_range(i,:)-L2_SD_range(refInd,:);
        
        L1_DD_Phase(k,:)=L1_SD_phase(i,:)-L1_SD_phase(refInd,:);
        L2_DD_Phase(k,:)=L2_SD_phase(i,:)-L2_SD_phase(refInd,:);
    end
end


%% Step # 5
% estimate the Wide lane ambiguity

ambEst_WL=WL_DD_Phase-L1_DD_Range/lambda_L12;
% plot these
figure,plot(round(ambEst_WL'))
grid on
title(' Wide Lane Ambiguity Estimates')


%% Step #6 "Integerize" the estimates by averaging and rounding
N_L12=round(mean(ambEst_WL'));

%% Step #7  Now estimate N_L1 using N_L12

est_NL1_NWL=(1/(alpha-1))*(-L1_DD_Phase/lambda_L1+alpha*L2_DD_Phase/lambda_L2)+(alpha/(alpha-1))*N_L12'*ones(1,length(L1_DD_Phase));
% plot these
figure,plot(round(est_NL1_NWL'))
title(' DD L1  Ambiguity Estimates using WL Ambiguity','FontSize',16,'FontWeight','bold')
xlabel(' Time (s)','FontSize',16,'FontWeight','bold')
ylabel(' L1 Ambiguity Estimate','FontSize',16,'FontWeight','bold')
grid on

N_L1_est=round((est_NL1_NWL'));

N_L1=round(mean(est_NL1_NWL'));





%% Step 8#

for i=1:1600
    
    
    % assume base-station position is well known ( usin the truth position here)
    dgps_XYZ_EST(:,i)=LLS_DGPS_L1(truthXYZ_R(:,1)',satsXYZ(1:nSat_U(i),:,i),...
    nSat_U(i),6,L1_DD_Phase(:,i)/lambda_L1,N_L1');
   
    % per epoch, real-time
    %dgps_XYZ_EST(:,i)=LLS_DGPS_L1(truthXYZ_R(:,1)',satsXYZ(1:nSat_U(i),:,i),...
     %   nSat_U(i),6,L1_DD_Phase(:,i)/lambda_L1,N_L1_est(i,:)');
    
    
    %% Note that that mathematics are the same if we use DD Range, just no ambiguity
    % note I am dividing Range by lambda_L1 and setting ambiguity to zero
    dgpsRange_XYZ_EST(:,i)=LLS_DGPS_L1(truthXYZ_R(:,1)',satsXYZ(1:nSat_U(i),:,i),...
        nSat_U(i),6,L1_DD_Range(:,i)/lambda_L1,zeros(size(N_L1')));
    
    % standard iono-free LLS with trop
   % [xyzEstPR(:,i),clockBiasEst]= LLSGPS(truthXYZ_R(:,1)',0,satsXYZ(1:nSat_U(i),:,i),...
    %    2.546*prDataP1_U(:,i)-1.546*prDataP2_U(:,i),nSat_U(i),0, 0,0,1);
    
    % standard iono-free LLS with no trop tro
    [xyzEstPR(:,i),clockBiasEst]= LLSGPS(truthXYZ_R(:,1)',0,satsXYZ(1:nSat_U(i),:,i),...
       2.546*prDataP1_U(:,i)-1.546*prDataP2_U(:,i),nSat_U(i),0, 0,0,0);
    
    %standard  LLS with no trop or iono
    %[xyzEstPR(:,i),clockBiasEst]= LLSGPS(truthXYZ_R(:,1)',0,satsXYZ(1:nSat_U(i),:,i),...
    % prDataP1_U(:,i),nSat_U(i),0, 0,0,0);
    
    dgps_enu_EST(:,i)=xyz2enu(dgps_XYZ_EST(:,i),truthXYZ_R(:,1));
    dgpsRange_enu_EST(:,i)=xyz2enu(dgpsRange_XYZ_EST(:,i),truthXYZ_R(:,1));
    lls_enu_EST(:,i)=xyz2enu(xyzEstPR(:,i),truthXYZ_R(:,1));
    truth_enu_U(:,i)=xyz2enu(truthXYZ_U(:,i),truthXYZ_R(:,1));
    
end

figure,subplot(311),plot(lls_enu_EST(1,:)-truth_enu_U(1,:),'b--');
hold on
plot(dgpsRange_enu_EST(1,:)-truth_enu_U(1,:),'m:','LineWidth',1);
plot(dgps_enu_EST(1,:)-truth_enu_U(1,:),'r','LineWidth',3);


grid on
xlabel('time (s)')
ylabel('East Error (m)')
legend('PR LLS','PR-DD DGPS','CP-DD DGPS')

subplot(312),plot(lls_enu_EST(2,:)-truth_enu_U(2,:),'b--');
hold on
plot(dgpsRange_enu_EST(2,:)-truth_enu_U(2,:),'m:','LineWidth',1);
plot(dgps_enu_EST(2,:)-truth_enu_U(2,:),'r','LineWidth',3);
grid on
xlabel('time (s)')
ylabel('North Error (m)')
legend('PR LLS','PR-DD DGPS','CP-DD DGPS')


subplot(313),plot(lls_enu_EST(3,:)-truth_enu_U(3,:),'b--');
hold on
plot(dgpsRange_enu_EST(3,:)-truth_enu_U(3,:),'m:','LineWidth',1);
plot(dgps_enu_EST(1,:)-truth_enu_U(1,:),'r','LineWidth',3);
grid on
xlabel('time (s)')
ylabel('Vertical Error (m)')
legend('PR LLS','PR-DD DGPS','CP-DD DGPS')

%% Alternative,Step # 8 now estimate N_L2 from N_L1 and N_L12 and use both N_L1 and N_L2 for positioning
N_L2=N_L12-N_L1;
disp(N_L2)





