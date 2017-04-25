% MAE 565 Artificial Intelligence Techniques in MAE - Demo
% ANN for Function Approximation - ANN Training in one Step

% RBF ANN - Number of neurons (centers) = number of training data points


% %close all
% clear all
% 
% % Generate training data (function to be approximated)
% inp1 = -2:0.2:2;                                 % input (xbar) training data
% for i = 1:length(inp1)
%     y(i) = (sin(inp1(i)))*exp(-(abs(inp1(i))));  % output training data
% end
% 
% figure,plot(inp1,y)
% title('Function To Be Approximated - Training Data')
% xlabel('Independent Variable x')
% ylabel('Function y=f(x)')
% grid
inp1=inp_train;
y=y6;
% Define the ANN parameters
 centers = inp1;          % Case #1: centers defined at the same location as the training data --> get an interpolation tool
%centers = inp1-0.025;   % Case #2: as many centers as data points but different location
sigma = 0.025;            % width of the Gaussian.  Will vary this parameter to 0.01, 0.1, 0.5, and 0.35


% Start the training process - only one iteration to determine the weights
% (See Section 4.9)
for i = 1:length(centers)
    for j = 1:length(inp1)
D(i,j) = abs(centers(i)-inp1(j));      % Compute the distance between centers and the training data points (Matrix D, Eqn. 4.73)
end
end

% Compute the values of the Gaussian function for distances between centers and points
% = output of the HL
PhiofD = 1/sqrt(2*pi*sigma^2)*exp(-D.^2/2/sigma^2);   % Eqn. 4.74  

% Compute the weights
Z = inv(PhiofD)*y';       % Eqn 4.76

% ANN validation process 
% Step #1  -  Test the ANN with the inputs of the training data

for i = 1:length(inp1)
    
% distance to centers
d2c = abs(centers-inp1(i));

% output of the HL
yHL = 1/sqrt(2*pi*sigma^2)*exp(-d2c.^2/2/sigma^2);

% output of ANN
YestO(i) = yHL*Z;

end

figure, plot(inp1,y,'k',inp1,YestO,'r')
title('ANN Response to Training Data')
legend('Training Data','ANN Estimation',2)
xlabel('Independent Variable x')
ylabel('Function y=f(x)')
grid

% Step #2  -  Valoidation - Test the ANN with different data but still produced by the
% same "function" 

% Generate validation data using better resolution (0.01)
% and expanding the domain to check if extrapolation is performed
% correctly

% Compute the validation data input
% inp1val = -4:0.01:4;  
% 
% % Compute the validation data output
% for i = 1:length(inp1val)
%     yval(i) = (sin(inp1val(i)))*exp(-(abs(inp1val(i))));
% end
inp1val=inp_valid;
yval=y6val;
% Compute the ANN estimation using the weights obtained through training
for i = 1:length(inp1val)

% distance to centers
d2c = abs(centers-inp1val(i));

% Output of the HL
yHL = 1/sqrt(2*pi*sigma^2)*exp(-d2c.^2/2/sigma^2);

% Output of ANN
YestO(i) = yHL*Z;

end

figure, plot(inp1val,yval,'k',inp1val,YestO,'r')
title('ANN Response to Validation Data')
legend('Validation Data','ANN Estimation',2)
grid

figure, plot(inp1val,yval-YestO,'k')
title('Error of the ANN Response to Validation Data Input')
xlabel('Independent Variable x')
ylabel('Estimation Error')
grid
