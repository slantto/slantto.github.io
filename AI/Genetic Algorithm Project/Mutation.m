function [ NewPop1 ] = Mutation(Current_Gen_Num, Mute_Ind,MaxGen)
% Non-Uniform Mutation
%   Detailed explanation goes here

%Individual = [];
UB = 56371; % Upper Bound
LB = 6871; % Lower Bound
SIZE = [size(Mute_Ind)];

NewPop1 = zeros(SIZE); % Initialize New Matrix

t = Current_Gen_Num;
T = MaxGen;
b = 0.07; % Don't know what this is. YOLO

for j = 1:SIZE(1)
    for i = 1:SIZE(2)
        Epsi = round(rand); % Randomly Generate 1 or 0
        if Epsi == 0;
            a = -1;
            y = UB - Mute_Ind(j,i);
        elseif Epsi == 1;
            a = 1;
            y = Mute_Ind(j,i) - LB;
        end
        r = rand();
        delta = y*(1-r^(1-(t/T))^(b));
        NewPop1(j,i) = Mute_Ind(j,i) + a*delta;
        
    end
end


end

