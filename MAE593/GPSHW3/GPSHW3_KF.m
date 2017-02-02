clear all

load('realData.mat')

P= diag([1,1,1,ones(1,numAmbStates)*100^2]);

% 
R = (.245^2)*(ones(numAmbStates,numAmbStates)+eye(numAmbStates));

% state transition matrix is just PRNentity
F= eye(numNonAmbStates+ numAmbStates);

% platform is moving up to ~1 m/s and ambigutities are constant
Q=blkdiag(eye(numNonAmbStates)*1^2, zeros(numAmbStates,numAmbStates)) ;

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

