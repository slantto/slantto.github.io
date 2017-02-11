%Fitness Parameters 

% ni = number of individuals
% delv = delta v (1,2,3,4)
%dv_tot = total delta v (to be compared to Hohmann transfer value)
%fit_dv = portion of fitness criterion dictated by quality of total delta v
%DVFIT = portion of universal fitness to be added for measurement of
%individual's fitness regarding Delta V
%tof = time of flight (to be compared to Hohmann transfer value)
%fit_tof = portion of fitness criterion dictated by quality of time of
%flight
%TOFFIT = portion of universal fitness to be added for measurement of
%individual's fitness regarding time of flight
%FITNESS = Individual's overall fitness on a scale of 0 to 1

function [DVFIT,TOFFIT,FITNESS] = fitnessfxn(numindividuals,delv1,delv2,delv3,delv4,tof)

for i=1:numindividuals
    
    dv_tot(i) = delv1(i) + delv2(i) + delv3(i) + delv4(i);
    
        if dv_tot(i) <= .9375
        fit_dv(i) = .1;
        
    elseif (dv_tot(i) > .9375 && dv_tot(i) <= 1.947)
        fit_dv(i) = .6;
        
    elseif (dv_tot(i) > 1.947 && dv_tot(i) <= 2.8845)
        fit_dv(i) = .9;
        
    elseif (dv_tot(i) > 2.8845 && dv_tot(i) <= 3.984)
        fit_dv(i) = .7;
        
    elseif dv_tot(i) > 3.894
        fit_dv(i) = .1;
        
    elseif dv_tot(i) == 1.947
        fit_dv(i) = 1;
    end
    
    DVFIT(i) = .8*fit_dv(i);
    %disp(DVFIT(i))
    
    if tof(i) <= 2.15112
        fit_tof(i) = .1;
        
    elseif (tof(i) > 2.15112 && tof(i) <= 3.5851)
        fit_tof(i) = .6;
        
    elseif (tof(i) > 3.5851 && tof(i) <= 10.2074)
        fit_tof(i) = .9;
        
    elseif (tof(i) > 10.2074 && tof(i) <= 168)
        fit_tof(i) = .7;
        
    elseif tof(i) > 168
        fit_tof(i) = .1;
        
    elseif tof(i) == 3.5852
        fit_tof(i) = 1;
    end
    
    TOFFIT(i) = .2*fit_tof(i);
   % disp(i)
    FITNESS(i) = DVFIT(i) + TOFFIT(i);
    
end