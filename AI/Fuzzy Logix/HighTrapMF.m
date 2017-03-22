function [ y ] = HighTrapMF( X, Points )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

a = Points(1);
b = Points(2);
c = Points(3);

y = zeros(1,length(X));

for i = 1:length(X)
    % a = -500
    % b = -200
    % c = -150
    if a >= X(i)
        y(i) = 0;
    elseif a <= X(i) && b >= X(i)
        y(i) = (X(i)-a)/(b-a);
    elseif b <= X(i) && c >= X(i)
        y(i) = 1;
        
    end
    
end

return

end
