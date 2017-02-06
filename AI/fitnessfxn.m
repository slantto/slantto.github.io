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

function [DVFIT,TOFFIT,FITNESS] = fitnessfxn(ni,delv1,delv2,delv3,delv4,tof)

for i=1:ni
    
    dv_tot(ni) = delv1(ni) + delv2(ni) + delv3(ni) + delv4(ni);
    
        if dv_tot(ni) <= .9375
        fit_dv(ni) = .1;
        
    elseif (dv_tot(ni) >= .9376 && dv_tot(ni) <= 1.947)
        fit_dv = .6;
        
    elseif (tof(ni) >= 1.948 && tof(ni) <= 2.8845)
        fit_dv(ni) = .8;
        
    elseif (dv_tot(ni) >= 2.8846 && dv_tot(ni) <= 3.984)
        fit_dv(ni) = .7;
        
    elseif dv_tot(ni) >= 3.895
        fit_dv(ni) = .1;
        
    elseif dv_tot(ni) == 1.947
        fit_dv(ni) = 1;
    end
    
    DVFIT(ni) = .2*fit_tof;
    
    
    if tof(ni) <= 2.15112
        fit_tof(ni) = .1;
        
    elseif (tof(ni) >= 2.15113 && tof(ni) <= 3.5851)
        fit_tof = .6;
        
    elseif (tof(ni) >= 3.5853 && tof(ni) <= 82.2074)
        fit_tof(ni) = .8;
        
    elseif (tof(ni) >= 82.2075 && tof(ni) <= 168)
        fit_tof(ni) = .7;
        
    elseif tof(ni) >= 168.0001
        fit_tof(ni) = .1;
        
    elseif tof(ni) == 3.5852
        fit_tof(ni) = 1;
    end
    
    TOFFIT(ni) = .2*fit_tof;
    
    FITNESS(ni) = DVFIT(ni) + TOFFIT(ni);
    
end