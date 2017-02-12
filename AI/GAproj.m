%Genetic Algorithm Project
clear
clc
%% Initialize first population
Rearth=6371; %km radius of earth
mu=3.986004415*10^5; %km^3/s^2 Gravitational parameter of earth
R0=5000+Rearth;%Initial orbit at 5000km altitude
Rf=20000+Rearth; %Desired orbit at 20000km altitude
lower=500+Rearth;%lower limit for transfer orbit radii
upper=50000+Rearth;%upper limit for transfer orbit radii
MaxGen=10000;

numindividuals=100;

delv1=zeros(1,numindividuals);
delv2=delv1;
delv3=delv1;
delv4=delv1;
R2=delv1;
R3=delv1;
tof=zeros(1,numindividuals);
for k=1:numindividuals
numVar=2;
Trans=lower+(upper-lower).*rand(1,numVar);
R2(k)=Trans(:,1);
R3(k)=Trans(:,2);

x=Rf/R0;
y=R2(k)/R0;
z=R3(k)/R0;

delv1(k)=abs(sqrt(mu/R0)*(sqrt(((2*y)/(1+y)))-1));
delv2(k)=abs(sqrt(mu/R0)*sqrt(1/y)*(sqrt((2*z)/(z+y))-sqrt(2/(1+y))));
delv3(k)=abs(sqrt(mu/R0)*sqrt(1/z)*(sqrt((2*y)/(z+y))-sqrt((2*x)/(z+x))));
delv4(k)=abs(sqrt(mu/R0)*sqrt(1/x)*(sqrt(((2*x)/(z+x)))-1));
TOF1=pi*sqrt(((R0+R2(k))^3)/(8*mu));
TOF2=pi*sqrt(((R2(k)+R3(k))^3)/(8*mu));
TOF3=pi*sqrt(((R3(k)+Rf)^3)/(8*mu));
tof(k)=TOF1+TOF2+TOF3;% time of flight in seconds
end
R2=R2';
R3=R3';
tof=tof'/3600;%puts time of flight in hours
delv=[delv1',delv2',delv3',delv4'];%all delta-v's are in km/s
dVtotes=delv1'+delv2'+delv3'+delv4';

Pop=horzcat(R2,R3);

for ijk=1:MaxGen
    %%
    if ijk>1
        R2=Pop(:,1);
        R3=Pop(:,2);
        for k=1:numindividuals
            x=Rf/R0;
            y=R2(k)/R0;
            z=R3(k)/R0;
            
            delv1(k)=abs(sqrt(mu/R0)*(sqrt(((2*y)/(1+y)))-1));
            delv2(k)=abs(sqrt(mu/R0)*sqrt(1/y)*(sqrt((2*z)/(z+y))-sqrt(2/(1+y))));
            delv3(k)=abs(sqrt(mu/R0)*sqrt(1/z)*(sqrt((2*y)/(z+y))-sqrt((2*x)/(z+x))));
            delv4(k)=abs(sqrt(mu/R0)*sqrt(1/x)*(sqrt(((2*x)/(z+x)))-1));
            TOF1=pi*sqrt(((R0+R2(k))^3)/(8*mu));
            TOF2=pi*sqrt(((R2(k)+R3(k))^3)/(8*mu));
            TOF3=pi*sqrt(((R3(k)+Rf)^3)/(8*mu));
            tof(k)=TOF1+TOF2+TOF3;% time of flight in seconds
        end
        tof=tof/3600;%puts time of flight in hours
        
        Pop=horzcat(R2,R3);
    end
    %% Fitness Evaluation and Selection
    %Call for fitness and roulette
    
    randomnumber = rand(numindividuals,1);
    newpop = zeros(numindividuals,2);
    
    
    
    [DVFIT,TOFFIT,FITNESS] = fitnessfxn(numindividuals,delv1,delv2,delv3,delv4,tof);
    
    totfit = sum(FITNESS); %Summing the fitness of all individuals
    
    prob = FITNESS/totfit; %Calculates probability of each individual
    
    
    
    roundp = round(prob,3);
    q(1)=prob(1);
    for i=2:numindividuals
        q(i)=sum(prob(1:i));
    end
    randomround = round(randomnumber,3);
    
    for i=1:numindividuals
        
        ind(i) = q(i);
        
        for j = 1:numindividuals
            
            if i == 100
                newpop(i,:)=Pop(i,:);
                
            elseif randomround(j)>q(i) && randomround(j)<=q(i+1)
                newpop(i,:)=Pop(i+1,:);
            else
                newpop(i,:)=Pop(i,:);
            end
        end
        
    end


%% Mutation and Crossover

N = numindividuals;
CrossRate = .8;
MuteRate = .1;
Population = newpop; % Population containing N individuals
Current_Gen_Num = ijk; % This number changes +1 for every generation

Num_Cross_Ind = floor(N*CrossRate); % # of Individuals to be Crossovered
Num_Mute_Ind = floor(N*MuteRate); % # of Individuals to be Mutated

Loc_Cross_Ind = randperm(N,Num_Cross_Ind); % Indices of the Individuals from the Population (Crossover)
Rows_Left = setdiff([1:N],Loc_Cross_Ind); % Remain indices that haven't been randomly selected
Loc_Mute_Ind = Rows_Left(:,randperm(N-Num_Cross_Ind,Num_Mute_Ind)); % Indices of the Individuals from the Population (Mutation)

Cross_Ind = Population(Loc_Cross_Ind,:); % These individuals will have the crossover function applied to them.
Mute_Ind = Population(Loc_Mute_Ind,:); % These individuals will have the mutation function applied to them                        
                         
Match = sum(ismember(Loc_Cross_Ind,Loc_Mute_Ind)); % Making sure Individuals selected from the Pop are not the same. Match should be 0





[ NewPop1 ] = Mutation(Current_Gen_Num, Mute_Ind,MaxGen); % Newly mutated individuals
[ NewPop2 ] = Crossover( Cross_Ind ); % New Individuals after crossover
[ NewPop3 ] = Population(setdiff(Rows_Left,Loc_Mute_Ind),:); % Individuals that went unchanged

NewPopFinal = [ NewPop1; NewPop2; NewPop3]; % Concatenated Population Matrix
Pop=NewPopFinal;
end