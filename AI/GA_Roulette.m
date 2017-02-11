%Call for fitness and roulette

randomnumber = rand(numindividuals,1);
newpop = zeros(numindividuals,3);

for i=1:numindividuals
    
    [DVFIT(i),TOFFIT(i),FITNESS(i)] = fitnessfxn(i,delv1,delv2,delv3,delv4,tof);
    
    totfit = sum(FITNESS); %Summing the fitness of all individuals
    
    prob(i) = FITNESS(i)/totfit; %Calculates probability of each individual
 
end

    roundp = round(prob,3);
    randomround = round(randomnumber,3);
    
    for i=1:numindividuals
        
        ind(i) = roundp(i);
        
        for j = 1:numindividuals
            if ind == randomround(j)
                ind = newpop(i);
            else 
            end
        end
                
    end
    
    