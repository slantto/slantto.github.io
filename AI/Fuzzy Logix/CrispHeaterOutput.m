function [ output_args ] = CrispHeaterOutput( FuzzyTempInput )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
Range = -5:.001:4.999

Zero = TrapMF(Range,[-1,-0.5,0.5,1]);
One = TrapMF(Range,[-1,-.25,.25,1]);
Two = TrapMF(Range,[.25,1.5,2.5,4]);

hold on
plot(Range,Zero,'k')
grid on
axis([-5 5 -.05 1.05])
end

