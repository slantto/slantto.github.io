function [ y ] = LowTrapMF( X, Points )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

a = Points(1);
b = Points(2);
c = Points(3);

y = zeros(1,length(X));

for i = 1:length(X)

if a <= X(i) && b >= X(i)
    y(i) = 1;
elseif b <= X(i) && c >= X(i)
    y(i) =abs((c-X(i)))/abs((b-c));
else
    y(i) = 0;
end

end
end
