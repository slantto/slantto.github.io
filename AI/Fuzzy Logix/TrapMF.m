function [ y ] = TrapMF( X, Points )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

a = Points(1);
b = Points(2);
c = Points(3);
d = Points(4);
y = zeros(1,length(X));

for i = 1:length(X)
    
    if X <= a
        y(i) = 0;
    elseif a <= X(i) && b >= X(i)
        y(i) = (X(i)-a)/(b-a);
    elseif b <= X(i) && c >= X(i)
        y(i) = 1;
    elseif c <= X(i) && d >= X(i)
        y(i) = (d-X(i))/(d-c);
    else
        y(i) = 0;
    end
    
end


plot([-10:.001:9.999],y)
end

