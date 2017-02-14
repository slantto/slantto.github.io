function [ NewPop1 ] = Muta(Current_Gen_Num, Mute_Ind,MaxGen)
% Non-Uniform Mutation
%   Detailed explanation goes here

%Individual = [];
UB = 56371; % Upper Bound
LB = 6871; % Lower Bound
SIZE = [size(Mute_Ind)];

NewPop1 = zeros(SIZE); % Initialize New Matrix

t = Current_Gen_Num;
T = MaxGen;
b = 0.05; % Don't know what this is. YOLO

for j = 1:SIZE(1)
    for i = 1:SIZE(2)
        Epsi = round(rand); % Randomly Generate 1 or 0
        if Epsi == 0 && Mute_Ind(j,i)<=UB-1000;
            y = 100+Mute_Ind(j,i);
        elseif Epsi == 1 && Mute_Ind(j,i)>=LB+1000;
            y = Mute_Ind(j,i) - 100;
        end
        NewPop1(j,i) = Mute_Ind(j,i);
        
    end
end


end
