%Fuzzy Logic Matrix Building

time = 2400; %Can be changed to however long we want to run this simulation

i = 1:time;

InputMatrix = zeros(5,5);
FuzzySet = zeros(2,2);
im1 = 1:5;
im2 = 1:5;
j = 1:25;
m = 1:4;
n = 1:2;
o = 1:2;

for i = 1:time
    
    [ FuzzyTempInput ] = TemperatureMF(OvenTempError);
    [ FuzzyTempRateInput ] = TemperatureRateMF(OvenTempRate);
    
    
    %Below is a loop that forms the Input Matrix based upon the minimum
    %preference. This can be changed to use any of the two others.
    for im1 = 1:5
       for im2 = 1:5
        if FuzzyTempInput(im2) <= FuzzyTempRateInput(im2)
            InputMatrix(im1,im2) = FuzzyTempInput(im2);
        else
            InputMatrix(im1,im2) = FuzzyTempRateInput(im2);
        end
       end
    end
    
   [row,col,v] = find(InputMatrix); %Finds non-zero values of the formed Matrix
   
   %Below loop forms smaller Fuzzy Set matrix
   for n = 1:2
       for o = 1:2
       FuzzySet(n,o) = v(m);
       end
   end
    
end
