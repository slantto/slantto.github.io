% MAE 565 Aritificial Neural Network Indiividual Assignment
%Sean Lantto
%701168294
%April 26,2017

% RBF ANN - Arbitrary number of neurons (centers)
for TrialCenter=1:2
    for TrialSigma=1:2
        
        %     close all
        clearvars -except TrialCenter TrialSigma
        load NN_proj_data_06
        
        inp1=inp_train;
        y=y6;
        
%         figure,plot(inp1,y)
%         title('Function To Be Approximated - Training Data')
%         xlabel('Independent Variable x')
%         ylabel('Function y=f(x)')
%         grid
        
        % Define the ANN parameters
        if TrialCenter == 1
            centers = inp1;
        elseif TrialCenter == 2
            centers = [-2:0.1:2];
        end
        
        if TrialSigma == 1
            sigma = 0.05;                    % width of the Gaussian
        elseif TrialSigma == 2
            sigma=[-0.05:0.1/length(centers):0.05];
            sigma=sigma(1:length(centers));
        end
        
        
        
        % Start the training process - only one iteration to determine the weights
        if TrialSigma == 1
            for i = 1:length(centers)
                for j = 1:length(inp1)
                    D(i,j) = abs(centers(i)-inp1(j));         % Compute the distance between centers and the training data points (Matrix D)
                    PhiofD(i,j) = 1/sqrt(2*pi*sigma^2)*exp(-D(i,j).^2/2/sigma^2);
                end
            end
        elseif TrialSigma == 2
            for i = 1:length(centers)
                for j = 1:length(inp1)
                    D(i,j) = abs(centers(i)-inp1(j));         % Compute the distance between centers and the training data points (Matrix D)
                    PhiofD(i,j) = 1/sqrt(2*pi*sigma(i).^2)*exp(-D(i,j).^2/2/sigma(i).^2);
                end
            end
        end
        % Compute the values of the Gaussian function for distances between centers and points
        % = output of the HL
        
        
        % Compute the weights
        Z(:,TrialCenter,TrialSigma) = pinv(PhiofD)'*y';
        
        
        % ANN validation process
        % Step #1  -  Test the ANN with the inputs of the training data
        
        for i = 1:length(inp1)
            
            % distance to centers
            d2c = abs(centers-inp1(i));
            
            % output of the HL
            if TrialSigma == 1
                yHL = 1/sqrt(2*pi*sigma^2).*exp(-d2c.^2/2/sigma^2);
            elseif TrialSigma == 2
                yHL = 1./sqrt(2*pi.*sigma.^2).*exp(-d2c.^2./2./sigma.^2);
            end
            % output of ANN
            YestO(i) = yHL*Z(:,TrialCenter,TrialSigma);
            
        end
        
        figure, plot(inp1,y,'k',inp1,YestO,'r')
        title('ANN Response to Training Data')
        legend('Training Data','ANN Estimation',2)
        xlabel('Independent Variable x')
        ylabel('Function y=f(x)')
        grid
        
        
        % Step #2  -  Test the ANN with different data but still produced by the
        % same "function"
        
        inp1val=inp_valid;
        yval=y6val;
        
        % Compute the ANN estimation using the weights obtained through training
        for i = 1:length(inp1val)
            
            % distance to centers
            d2c = abs(centers-inp1val(i));
            
            % output of the HL
            if TrialSigma == 1
                yHL = 1/sqrt(2*pi*sigma^2).*exp(-d2c.^2/2/sigma^2);
            elseif TrialSigma == 2
                yHL = 1./sqrt(2*pi.*sigma.^2).*exp(-d2c.^2./2./sigma.^2);
            end
            % output of ANN
            YestO(i) = yHL*Z(:,TrialCenter,TrialSigma);
            
        end
        
        TrialFile = ['RBF_ANN_CenterCase' num2str(TrialCenter) '_SigmaCase' num2str(TrialSigma)]
        save(TrialFile);
        figure, plot(inp1val,yval,'k',inp1val,YestO,'r')
        title('ANN Response to Validation Data')
        legend('Validation Data','ANN Estimation')
        xlabel('Independent Variable x')
        ylabel('Function y=f(x)')
        grid
        
    end
end

clear
figure;

for TrialSigma = 1:2
    for TrialCenter = 1:2
        TrialFile = ['RBF_ANN_CenterCase' num2str(TrialCenter) '_SigmaCase' num2str(TrialSigma)]
        load(TrialFile);
        
        plot(inp1val,yval-YestO); hold on
        title('Error of the ANN Response to Validation Data Input')
        xlabel('Independent Variable x')
        ylabel('Estimation Error')
        grid on
        
    end
end