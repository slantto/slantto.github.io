function [ FuzzyTempRateInput ] = TemperatureRateMF( OvenTempRate )
% Linguistic Values: Very Negative, Negative, Zero, Positive, Very Positive
% -----------------------
%
% Overall Temperature Acceleration Range Will be +- 10 deg/sec


% Range = -10:0.001:9.999;
% VN = LowTrapMF(Range,[-10,-4,-2]);
% N = TrapMF(Range,[-4,-2.5,-1.5,-.25]);
% Zero = TrapMF(Range,[-1,-.25,.25,1]);
% P = TrapMF(Range,[.25,1.5,2.5,4]);
% VP = HighTrapMF(Range,[2,4,10]);
% 
% figure
% hold on
% plot(Range,VN,'k')
% plot(Range,N,'k')
% plot(Range,Zero,'k')
% plot(Range,P,'k')
% plot(Range,VP,'k')
% xlabel('Temperature Acceleration (\circc/sec)')
% ylabel('Fuzzy Set Value')
% title('Temperature Acceleration Membership Function')
% grid minor
% axis([-10 10 -.05 1.05])
%
% save TempRateMF VN N Zero P VP Range

Scale = 5;
load TempRateMF
OvenTempRate = OvenTempRate*Scale;
TempRateIndex = round(1000*OvenTempRate + 10000);
FuzzyTempRateInput = [VN(TempRateIndex),N(TempRateIndex),Zero(TempRateIndex),P(TempRateIndex),VP(TempRateIndex)];
end