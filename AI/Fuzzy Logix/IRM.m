function [ y ] = InferenceRuleMatrix( X, Points )

for i = 1:time
    
    % Access Membership Functions
    [ FuzzyTempInput ] = TemperatureMF(OvenTempError);
    [ FuzzyTempRateInput ] = TemperatureRateMF(OvenTempRate);
    
    % Create Inference Rule Matrix
    for im1 = 1:5;
        for im2 = 1:5;
            
            IRM(im1,im2) = min(FuzzyTempRateInput(im1),FuzzyTempInput(im2));
            
        end
    end 
end