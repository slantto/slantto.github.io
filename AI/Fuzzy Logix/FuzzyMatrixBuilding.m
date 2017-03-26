%Fuzzy Logic Matrix Building
OvenTempError = -20;
OvenTempRate = 2;
time = 1; %Can be changed to however long we want to run this simulation


IRM = zeros(5,5);
im1 = 1:5;
im2 = 1:5;


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

%------------------------------
% Map Heater IRM to Fuzzy CMD
Heater_Off = [IRM(1,1),IRM(2,1),IRM(3,1),IRM(3,2),IRM(3,3),IRM(4,1),...
    IRM(4,2),IRM(4,3),IRM(5,1),IRM(5,2),IRM(5,3)];
Heater_VeryLow = [IRM(2,2)];
Heater_Low = [IRM(1,2),IRM(2,3),IRM(5,4)];
Heater_High = [IRM(3,4),IRM(4,4)];
Heater_VeryHigh = [IRM(1,3),IRM(1,4),IRM(1,5),IRM(2,4),IRM(2,5),...
    IRM(3,5),IRM(4,5),IRM(5,5)];

%------------------------------
% Remove Zero Elements
Heater_Off(Heater_Off==0)=[];
Heater_VeryLow(Heater_VeryLow==0) = [];
Heater_Low(Heater_Low==0) = [];
Heater_High(Heater_High==0) = [];
Heater_VeryHigh(Heater_VeryHigh==0) = []; 

%------------------------------
% Map Fan IRM to Fuzzy CMD
Fan_Off = [IRM(1,3),IRM(1,4),IRM(1,5),IRM(2,3),IRM(2,4),IRM(2,5),...
    IRM(3,3),IRM(3,4),IRM(3,5),IRM(4,4),IRM(4,5),IRM(5,4),IRM(5,5)];
Fan_Low = [IRM(1,1),IRM(1,2),IRM(2,1)];
Fan_Medium = [IRM(2,2),IRM(3,2),IRM(4,3)];
Fan_High = [IRM(3,1),IRM(4,1),IRM(4,2),IRM(5,1),IRM(5,2),IRM(5,3)];

%------------------------------
% Remove Zero Elements
Fan_Off(Fan_Off==0) = [];
Fan_Low(Fan_Low==0) = [];
Fan_Medium(Fan_Medium==0) = [];
Fan_High(Fan_High==0) = [];

% Some sort of Area equation here.


Centr1 = [0 1 2 3 4];
b = 1;
a1 = [-0.5*Heater_Off+1];
a2 = [-0.5*Heater_VeryLow+1];
a3 = [-0.5*Heater_Low+1];
a4 = [-0.5*Heater_High+1];
a5 = [-0.5*Heater_VeryHigh+1];

Area1 = 0.5.*(a1+b).*Heater_Off;
Area2 = 0.5.*(a2+b).*Heater_VeryLow;
Area3 = 0.5.*(a3+b).*Heater_Low;
Area4 = 0.5.*(a4+b).*Heater_High;
Area5 = 0.5.*(a5+b).*Heater_VeryHigh;

CrispHeaterOutput = sum([Area1*Centr1(1),Area2*Centr1(2),Area3*Centr1(3),Area4*Centr1(4),Area5*Centr1(5)])/...
    sum([Area1,Area2,Area3,Area4,Area5]);

Centr2 = [0 1 2 3];
a11 = [-0.5*Fan_Off+1];
a22 = [-0.5*Fan_Low+1];
a33 = [-0.5*Fan_Medium+1];
a44 = [-0.5*Fan_High+1];

Area11 = 0.5.*(a11+b).*Fan_Off;
Area22 = 0.5.*(a22+b).*Fan_Low;
Area33 = 0.5.*(a33+b).*Fan_Medium;
Area44 = 0.5.*(a44+b).*Fan_High;

CrispFanOutput = sum([Area11*Centr1(1),Area22*Centr1(2),Area33*Centr1(3),Area44*Centr1(4)])/...
    sum([Area1,Area2,Area3,Area4]);


x = [0 3];
y = [5 50];
c = [[1;1] x(:)]\y(:);
Slope = c(2)
Intercept = c(1)
