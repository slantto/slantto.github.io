% Plotting

% Vector Comes in at length = 3001
if length(ActualOvenTemperature) > 3000
ActualOvenTemperature(3001:end) = [];
end
if length(OvenTempError) > 3000
OvenTempError(3001:end) = [];
end

figure
plot((TimeForTemp/60),TempControl,'b',(TimeForTemp/60),ActualOvenTemperature,'r')
xlabel('Time (min)')
ylabel('Temperature (\circc)')
title('Thermostat Temperature vs. Actual Oven Temperature')
grid minor
legend('Set Temperature','Oven Temperature')



figure
plot((TimeForTemp/60),OvenTempError)
xlabel('Time (min)')
ylabel('Temperature (\circc)')
title('Thermostat Temperature Error vs. Time')
grid minor
axis([0 50 -150 200])






