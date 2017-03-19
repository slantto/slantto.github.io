%   MAE 565 Fuzzy Logic Script
% -------------------------------
% Problem constant
% -------------------------------
% converst radians to degrees
r2d = 180/pi;
% -------------------------------
% Define the Oven geometry
% -------------------------------
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

windowArea = htWindows*widWindows;
wallArea = 2*lenOven*htOven + 2*widOven*htOven + lenOven*htOven;
% -------------------------------
% Define the type of insulation used
% -------------------------------
% Glass wool in the walls, 0.2 m thick
% k is in units of J/sec/m/C - convert to J/hr/m/C multiplying by 3600
kWall = 0.038*3600;   % hour is the time unit
LWall = .2;
RWall = LWall/(kWall*wallArea);
% Glass windows, 0.01 m thick
kWindow = 0.78*3600;  % hour is the time unit
LWindow = .01;
RWindow = LWindow/(kWindow*windowArea);
% -------------------------------
% Determine the equivalent thermal resistance for the whole building via
% conduction. How do I add in the resistance from convection?
% -------------------------------
R_Cond = RWall*RWindow/(RWall + RWindow);

h_inf = 5.0;
totalArea = windowArea + wallArea;
R_Conv = 1/(h_inf*totalArea);
% c = cp of air (273 K) = 1005.4 J/kg-K
c = 1005.4;
% -------------------------------
% Determine total internal air mass = M
% -------------------------------
% Density of air at sea level = 1.2250 kg/m^3
densAir = 1.2250;
M = (lenOven*widOven*htOven)*densAir;
% -------------------------------














