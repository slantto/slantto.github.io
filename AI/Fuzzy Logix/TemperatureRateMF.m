function [ FuzzyTempRateInput ] = TemperatureRateMF( OvenTempRate )


Range = -10:0.001:9.999;



% VeryLow = LowTrapMF(Range,[-10,-4,-2]);
% Low = TrapMF(Range,[-4,-2.5,-1.5,-.25]);
% Zero = TrapMF(Range,[-1,-.25,.25,1]);
% High = TrapMF(Range,[.25,1.5,2.5,4]);
% VeryHigh = HighTrapMF(Range,[2,4,10]);
% save TempRateMF VeryLow Low Zero High VeryHigh
load TempRateMF
% figure
% hold on
% plot(Range,VeryLow,'k')
% 
% plot(Range,Low,'k')
% 
% plot(Range,Zero,'k')
% 
% plot(Range,High,'k')
% 
% plot(Range,VeryHigh,'k')
% grid on
% axis([-10 10 -.05 1.05])


TempRateIndex = round(1000*OvenTempRate + 10000);
disp(TempRateIndex)
FuzzyTempRateInput = [VeryLow(TempRateIndex),Low(TempRateIndex),Zero(TempRateIndex),High(TempRateIndex),VeryHigh(TempRateIndex)];

end