% MAE 565 Artificial Intelligence Techniques in MAE - Demo
% ANN for Function Approximation - Backpropagation Training Exercise 

% close all
clear all

% Define the ANN parameters

Ni = 1;                   % number of inputs
Nh = 10;                  % number of neurons in the HL
eta = 0.1;                % learning rate
Ntot = 10;                % total number of iterations for training
                          % vary this parameter: 10, 50, 100, 200, 500

% Initialize the weights
% weights of the neurons in the HL (for one input)
W = [0.1 0.2 0.1 0.2 0.1 0.2 0.1 0.2 0.1 0.2]';      % Initial guess for Eqn 4.52   
% W=[0.1 0.2 0.1 0.2 0.1 0.2 0.1 0.2 0.1 0.2 0.1 0.2 0.1 0.2 0.1 0.2 0.1 0.2 0.1 0.2]';
% weights of the neurons in the output layer (one neuron in the output
% layer)
Z = [0.2 0.1 0.25 0.1 0.1 0.1 0.25 0.1 0.1 0.25];    % Initial guess for Eqn 4.54    
% Z=[0.2 0.1 0.25 0.1 0.1 0.1 0.25 0.1 0.1 0.25 0.2 0.1 0.25 0.1 0.1 0.1 0.25 0.1 0.1 0.25];

% Generate training data (function to be approximated)
inp1 = -2:0.2:2;      % input (xbar) training data
cntrs = inp1;         % centers defined at the same location as the training data --> get an interpolation tool

nit=length(inp1);

% Calculate input/output pairs for training data (function to be approximated)
for i = 1:length(inp1)  % 21 points
    y(i) = (sin(inp1(i)))*exp(-(abs(inp1(i))));   % output training data
end


% Start the training process (see Section 4.8)

% Perform a prescribed number of exposures to the training data (iterations
% for training)
for itot = 1:Ntot      
% expose the ANN to each of the pairs inputs/outputs
for i = 1:nit   
% activation values of the HL neurons (Nh x 1 vector)
XstarH = W*inp1(i);       % Eqn. 4.58
% activation function is a bipolar sigmoid (tanh)
psiH = (exp(XstarH)-exp(-XstarH))./(exp(XstarH)+exp(-XstarH));    % Eqn. 4.59
YestH = psiH;             % output of the HL neurons
XstarO = Z*YestH;         % activation value of the output neuron 

% activation function of the output neuron is also a bipolar sigmoid (tanh)
psiO = (exp(XstarO)-exp(-XstarO))./(exp(XstarO)+exp(-XstarO));    % Eqn. 4.61

YestO = psiO;             % output of the output neuron
errO = y(i)-YestO;        % error    % Eqn. 4.63

psiDerO = 1-psiO.^2;      % the derivative of bipolar sigmoid (output layer)
deltaO = psiDerO*errO;    % error signal    % Eqn. 4.65

Z = Z+eta*deltaO.*YestH'; % Update the weights of the output neuron  - Eqn. 4.66

psiDerH = 1-psiH.^2;      % the derivative of bipolar sigmoid (HL)
deltaH = psiDerH.*Z'*deltaO;  % error signal propagated at the level of HL

W = W+eta*deltaH*inp1(i); % Update the weights of the HL neurons - Eqn. 4.69

end
itot
end
% End of the back-propagation training process

% Test the ANN with the inputs of the training data (this is a good
% verification step, but it is NOT validation, it is NOT ENOUGH)
for i = 1:nit
% activation values of the HL neurons
XstarH = W*inp1(i);
% output of the HL neurons
psiH = (exp(XstarH)-exp(-XstarH))./(exp(XstarH)+exp(-XstarH));
YestH = psiH;
% activation value of the output neuron
XstarO = Z*YestH;
% output of the output neuron = approximation of the given function
psiO = (exp(XstarO)-exp(-XstarO))./(exp(XstarO)+exp(-XstarO));
YestO(i) = psiO;
end

figure, plot(inp1,y,'k',inp1,YestO,'r')
title('ANN Response to Training Data Input')
grid
xlabel('Independent Variable x')
ylabel('Function f(x)')
legend('Training Data','ANN Estimation')

% ANN validation process 

% Generate validation data using better resolution (0.01 instead of
% 0.05) and expanding the domain to check if extrapolation is performed
% correctly

% Calculate the validation data input
inp1val = -4:0.01:4;  

% Calculate the validation data output
for i = 1:length(inp1val)
    yval(i) = (sin(inp1val(i)))*exp(-(abs(inp1val(i))));
end


% Calculate the ANN estimation using the weights obtained through training
for i = 1:length(inp1val)

XstarH = W*inp1val(i);    % HL activation values
% HL outputs
psiH = (exp(XstarH)-exp(-XstarH))./(exp(XstarH)+exp(-XstarH));
YestH = psiH;
% output neuron activation value
XstarO = Z*YestH;
% ANN estimation
psiO = (exp(XstarO)-exp(-XstarO))./(exp(XstarO)+exp(-XstarO));
YestO(i) = psiO;
end

figure, plot(inp1val,yval,'k',inp1val,YestO,'r')
title('ANN Response to Validation Data Input')
grid
xlabel('Independent Variable x')
ylabel('Function f(x)')
legend('Validation Data','ANN Estimation')

figure, plot(inp1val,yval-YestO,'k')
title('Error of the ANN Response to Validation Data Input')
grid
xlabel('Independent Variable x')
ylabel('Estimation Error')
