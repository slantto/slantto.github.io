function [ NewPop2 ] = Crossover( Cross_Ind )
% Crossover Function
% Takes the last three terms of the row and 
% Switches with its neighbor
%
%


SIZE = [size(Cross_Ind)];
NewPop2 = zeros(SIZE);

for i = 1:2:SIZE(1)
    NewPop2([i,i+1],:) = [Cross_Ind(i,1) Cross_Ind(i+1,2);Cross_Ind(i+1,1) Cross_Ind(i,2)];
    
end

end

