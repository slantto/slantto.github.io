function [p_min, iter, f]=genetic_algorithm (func, numMaxInd, numF, numMaxGen, pRepr, pRecom, pMut, pImm, numVariabili, limLow, limUp, tolerance)

% A genetic algorithm is a search heuristic that mimics the process of 
% natural evolution. It is used to generate useful solutions to 
% optimization and search problems. Genetic algorithms belong to the larger
% class of evolutionary algorithms, which generate solutions to 
% optimization problems using techniques inspired by natural evolution, 
% such as inheritance, mutation, selection, and crossover.
%
% Output variables:
% - p_min: it's the minimum point of the objective function;
% - iter: it's the final iteration number;
% - f: it's the objective function value in the minimum point.
%
% Input variables:
% - func: it's the handle of the objective function to minimize (example: f_obj=@(x) function(x) where x is the variables vector); 
% - numMaxInd: Number of individuals (number of initial points);
% - numF: Number of sons for each generation;
% - numMaxGen: Max number of generations (number of max iterations);
% - pRepr: Reproduction probability;
% - pRecom: Ricombination probability;
% - pMut: Mutation probability;
% - pImm: Immigration probability;
% - numVariabili: it's the number of the function variables.
% - limLow: it's the low bound for the initial points;
% - limUp: it's the high bound for the initial points;
% - tolerance: it's the tolerance error for the stop criterion.

fOb=func;   % Objective Function (Function to minimize)
numInd=numMaxInd;   % Number of individuals (number of initial points)
numIter=numMaxGen;  % Number of generations (number of max iterations)
numFigliTot=numF;   % Number of sons for each generation.

% The sum of the probabilities of the mutation operations must be 1.
pRep=pRepr;     %Reproduction probability
pRec=pRecom;    %Ricombination probability

pM=pMut;    %Mutation probability
pI=pImm;    %Immigration probability

numVar=numVariabili;    %Number of function variables
low=limLow;
up=limUp;

% Initializes the initial population
ind=low+(up-low).*rand(numInd, numVar);

iter=1;

tolStop=tolerance;
stop=1;
while(iter<=numIter && stop>tolStop)
    
    %Calculates the population ranking
sumValF=0;
for l=1:numInd
    valF(l,1)=fOb(ind(l,:));
    sumValF=sumValF+(valF(l,1))^-1;
end
[valFord,Ind]=sort(valF,1,'ascend');
indOrd=ind(Ind,:);

for l=1:numInd
    rank(l,1)=(valFord(l,1))^-1/sumValF;
end
    
    numFigli=1;
    numFigli=numFigli+1;
    while (numFigli<=numFigliTot)
        p=rand();   % Probability to choose one genetic operation
            if p>=0 && p<pRep
               % disp('riprod')
                % Reproduction
                % Choose the parents with the wheel of fortune
                gen=rand();
                k=1;
                r=0;
                while(1)
                    r=r+rank(k,1);
                    if gen<=r;
                        break
                    else
                        k=k+1;
                    end
                end
            newGen(numFigli,:)=indOrd(k,:);
            numFigli=numFigli+1;
            else if p>=pRep && p<(pRec+pRep)
                % Ricombination  
               % disp('ricomb')
                minimo=low-1;
                mass=up+1;
                while(minimo<low || mass>up)
                    for h=1:2
                        gen=rand();
                        k=1;
                        r=0;
                        while(1)
                            r=r+rank(k,1);
                            if gen<=r;
                                break
                            else
                                k=k+1;
                            end
                        end
                    recomb(h,1)=k;
                    end
                alpha=normrnd(0.8,0.5);     % Calculates the alpha coefficient with a gaussian distribution with average = 0.8 and sigma = 0.5
                newI(1,:)=alpha.*indOrd(recomb(1,1),:)+(1-alpha).*indOrd(recomb(2,1),:);
                newI(2,:)=alpha.*indOrd(recomb(2,1),:)+(1-alpha).*indOrd(recomb(1,1),:);
                minimo=min(min(newI(1,:)),min(newI(2,:)));
                mass=max(max(newI(1,:)),max(newI(2,:)));
                end                
                newGen(numFigli,:)=newI(1,:);
                newGen(numFigli+1,:)=newI(2,:);               
                numFigli=numFigli+2;
                else if p>=(pRec+pRep) && p<(pRec+pRep+pM)
                % Mutation
               % disp('mutaz')
                minimo=low-1;
                mass=up+1;
                while(minimo<low || mass>up)
                    gen=rand();
                    k=1;
                    r=0;
                while(1)
                    r=r+rank(k,1);
                    if gen<=r;
                        break
                    else
                        k=k+1;
                    end
                end                
                    newI=indOrd(k,:)+normrnd(0,0.8,1,numVar);    %Add to genotype random values ??created by considering a Gaussian distribution with zero mean and variance 0.8 
                    minimo=min(newI);
                    mass=max(newI);
                end
                newGen(numFigli,:)=newI(1,:);
                numFigli=numFigli+1;
                    else if p>=(pRec+pRep+pM) && p<(pRec+pRep+pM+pI)
                % Immigration
               % disp('imm')
               for l=1:50
                newGen(numFigli,:)=low+(up-low).*rand(1, numVar);
                numFigli=numFigli+1;
               end
                        end
                    end
                end
            end
    end
    %Calculation of the best individuals that will be part of the new population
    indTot=[ind;newGen];
    for l=1:size(indTot,1)
        valFtot(l,1)=fOb(indTot(l,:));
    end
    [valFnewOrd,IndNew]=sort(valFtot,1,'ascend');
    indOrdNew=indTot(IndNew,:);
   % disp('new gen')
    ind=indOrdNew(1:numInd,:);
    iter=iter+1;
    
    %stop=abs(f(1)-f(numInd));
    stop=valFnewOrd(1);
end

f=valFnewOrd(1:numInd,:);
p_min=ind;