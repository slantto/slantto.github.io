clear all
clc

%   MAE 565 Fuzzy Logic Script
% -------------------------------
% Problem constants
% -------------------------------

% -------------------------------
% Define the Oven geometry
%

% Oven length = 1 m
lenOven = 1;
% House width = 1 m
widOven = 1;
% House height = 1 m
htOven = 1;
% Height of windows = .5 m
htWindows = .5;
% Width of windows = .5 m
widWindows = .5;

windowArea = htWindows*widWindows; % meters^2
wallArea = 2*lenOven*htOven + 2*widOven*htOven + lenOven*htOven; % meters^2

% -------------------------------
% Define the type of insulation used
% 

kWall = 20.00;   % W/m*K
LWall = .01;
RWall = LWall/(kWall*wallArea);

kWindow = 0.78;
LWindow = .01;
RWindow = LWindow/(kWindow*windowArea);

% -------------------------------
% Determine the equivalent thermal resistance for the whole building via
% conduction. The walls & windos are in parallel and the convection is
% considered in series.
% 

R_Cond = RWall*RWindow/(RWall + RWindow);

h_inf = 5.0;
Area = windowArea + wallArea;

R_Conv = 1/(h_inf*Area);

% c = cp of air (273 K) = 1005.4 J/kg-K
c_Air = 1005.4;
% cp of stainless steel J/kg/K
c_SS = 500.0;

% -------------------------------
%

densAir = 1.2250;
M_Air = (lenOven*widOven*htOven)*densAir;
M_SS = 12; %kg

% -------------------------------
% Input Variables
% 

M_cp = (M_Air*c_Air+M_SS*c_SS);
TimeForTemp = 1:3000;
TempControl = zeros(1,3000);

for i = 1:3000
    if i <= 600
        TempControl(i) = 175;
    elseif i > 600 && i <= 1000
        TempControl(i) = 220;
    elseif i > 1000 && i <= 1300
        TempControl(i) = 150;
    else
        TempControl(i) = 20;
    end
end





