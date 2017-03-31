function [ FuzzyTempInput ] = TemperatureMF( OvenTempError )
% Linguistic Values: Very Negative, Negative, Zero, Positive, Very Positive
% -----------------------
%
% Overall Temperature Error Range Will be +- 250c

% Range = -250:0.01:249.99;
% VN = LowTrapMF(Range,[-250,-7,-5]);
% N = TrapMF(Range,[-7,-5,-3,0]);
% Zero = TrapMF(Range,[-1,-0.5,0.5,1]);
% P = TrapMF(Range,[0,3,5,7]);
% VP = HighTrapMF(Range,[5,7,250]);
% 
% figure
% hold on
% plot(Range,VN,'k')
% plot(Range,N,'k')
% plot(Range,Zero,'k')
% plot(Range,P,'k')
% plot(Range,VP,'k')
% grid minor
% xlabel('Temperature Error (\circc)')
% ylabel('Fuzzy Set Value')
% title('Temperature Error Membership Function')
% axis([-10 10 -.05 1.05])
% 
% save TempMF VN N Zero P VP Range

load TempMF
TempIndex = round(100*OvenTempError + 25000);
FuzzyTempInput = [VN(TempIndex),N(TempIndex),Zero(TempIndex),P(TempIndex),VP(TempIndex)];
end

