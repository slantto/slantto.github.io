function [ y ] = FuzzyLogicThermostat( u )

% Assign Variables
OvenTempError = u(1);
OvenTempRate = u(2);

%------------------------------
% Fuzzification: Crisp Input => Fuzzy Input
%

[ FuzzyTempInput ] = TemperatureMF(OvenTempError);
[ FuzzyTempRateInput ] = TemperatureRateMF(OvenTempRate);

%
%------------------------------
% Inference Rule Matrix: 
%

IRM = InferenceRuleMatrix( FuzzyTempInput, FuzzyTempRateInput );

%
%------------------------------
% Defuzzification: Fuzzy Output => Crisp Output
%

[CrispHeaterOutput] = HeaterCommand( IRM );
[CrispFanOutput] = FanCommand( IRM );

y = [ CrispHeaterOutput, CrispFanOutput ];
end

