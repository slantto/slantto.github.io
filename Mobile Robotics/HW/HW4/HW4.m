%Problem 3 
%Likelihood of a coin landing heads up after observing HH
%Heads = Ph
%Tails = 1-Ph
clear 
close all
clc

Ph=0:0.1:1;

%part 1 HH
L=Ph.^2;
figure;
plot(Ph,L)
xlabel('Ph')
ylabel('Likelihood')
title('HH')

%Part 2 HHT
L=(Ph.^2).*(1-Ph);
figure;
plot(Ph,L)
xlabel('Ph')
ylabel('Likelihood')
title('HHT')


%Problem 4
%Coin type estimator
clear

%set up initial parameters
PrFC=1/3;
PrTC=1/3;
PrHC=1/3;
PrhFC=0.5;PrtFC=0.5;
PrhTC=0.4;PrtTC=0.6;
PrhHC=0.6;PrtHC=0.4;
coin=round(rand(400,1));
%Tails = 0,Heads = 1
for i=1:length(coin)
    if coin(i) == 1
        PrFCh=PrhFC*PrFC;
        PrTCh=PrhTC*PrTC;
        PrHCh=PrhHC*PrHC;
        normalize=1/(PrFCh+PrTCh+PrHCh);
        PrFC=PrFCh*normalize;
        PrTC=PrTCh*normalize;
        PrHC=PrHCh*normalize;
    elseif coin(i) == 0
        PrFCh=PrtFC*PrFC;
        PrTCh=PrtTC*PrTC;
        PrHCh=PrtHC*PrHC;
        normalize=1/(PrFCh+PrTCh+PrHCh);
        PrFC=PrFCh*normalize;
        PrTC=PrTCh*normalize;
        PrHC=PrHCh*normalize;
    end
    FC(i)=PrFC;
    TC(i)=PrTC;
    HC(i)=PrHC;
end

figure;
plot(1:length(coin),FC,1:length(coin),TC,1:length(coin),HC)
legend('FC','TC','HC')
xlabel('Number of flips')
ylabel('probability')

figure;
plot(1:length(coin),coin)
xlabel('Number of flips')
ylabel('Heads(1) or Tails(0)')
