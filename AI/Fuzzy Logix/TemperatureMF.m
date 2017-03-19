function [ FuzzyTempInput ] = TemperatureMF( OvenTempError )
% Linguistic Values
% -----------------------
% Very Low; Low; Set Point; High; Very High
%
% Overall Temperature Error Range Will be +- 


Range = -10:.001:9.999;
Scale1 = 35;


VeryLow = LowTrapMF(Range,[-10,-4,-2]);
Low = TrapMF(Range,[-4,-2.5,-1.5,-.25]);
Zero = TrapMF(Range,[-1,-.25,.25,1]);
High = TrapMF(Range,[.25,1.5,2.5,4]);
VeryHigh = HighTrapMF(Range,[2,4,10]);

hold on
plot(Range,VeryLow,'k')

plot(Range,Low,'k')

plot(Range,Zero,'k')

plot(Range,High,'k')

plot(Range,VeryHigh,'k')
grid on
axis([-10 10 -.05 1.05])

OvenTempError = OvenTempError/Scale1;
TempIndex = round(1000*OvenTempError + 10000);

FuzzyTempInput = [VeryLow(TempIndex),Low(TempIndex),Zero(TempIndex),High(TempIndex),VeryHigh(TempIndex)];
end

