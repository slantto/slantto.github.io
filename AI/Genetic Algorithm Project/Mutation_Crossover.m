% Of the entire population, 80 % will crossover
% and 10 % will mutate

% Assuming Population is a N by 5 matrix where N
% is equal to the number of individuals
% for example, N = 50

N = 50;
CrossRate = .8;
MuteRate = .1;
Population = rand(N,5); % Population containing N individuals
Current_Gen_Num = 1; % This number changes +1 for every generation

Num_Cross_Ind = floor(N*CrossRate); % # of Individuals to be Crossovered
Num_Mute_Ind = floor(N*MuteRate); % # of Individuals to be Mutated

Loc_Cross_Ind = randperm(N,Num_Cross_Ind); % Indices of the Individuals from the Population (Crossover)
Rows_Left = setdiff([1:N],Loc_Cross_Ind); % Remain indices that haven't been randomly selected
Loc_Mute_Ind = Rows_Left(:,randperm(N-Num_Cross_Ind,Num_Mute_Ind)); % Indices of the Individuals from the Population (Mutation)

Cross_Ind = Population(Loc_Cross_Ind,:); % These individuals will have the crossover function applied to them.
Mute_Ind = Population(Loc_Mute_Ind,:); % These individuals will have the mutation function applied to them                        
                         
Match = sum(ismember(Loc_Cross_Ind,Loc_Mute_Ind)); % Making sure Individuals selected from the Pop are not the same. Match should be 0





[ NewPop1 ] = Mutation(Current_Gen_Num, Mute_Ind); % Newly mutated individuals
[ NewPop2 ] = Crossover( Cross_Ind ); % New Individuals after crossover
[ NewPop3 ] = Population(setdiff(Rows_Left,Loc_Mute_Ind),:); % Individuals that went unchanged

NewPopFinal = [ NewPop1; NewPop2; NewPop3]; % Concatenated Population Matrix
















