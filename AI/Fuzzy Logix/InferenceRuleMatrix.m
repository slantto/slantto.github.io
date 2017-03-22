function [ IRM ] = InferenceRuleMatrix( FuzzyTempInput, FuzzyTempRateInput )



    % Create Inference Rule Matrix
    IRM = zeros(5,5);
    for im1 = 1:5;
        for im2 = 1:5;
            
            IRM(im1,im2) = min(FuzzyTempRateInput(im1),FuzzyTempInput(im2));
            
        end
    end 



end

