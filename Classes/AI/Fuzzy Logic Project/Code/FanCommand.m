function [CrispFanOutput] = FanCommand( IRM )

%------------------------------
% Map Fan IRM to Fuzzy CMD
%

Fan_Off = [IRM(1,4),IRM(1,5),IRM(2,4),IRM(2,5),...
    IRM(3,4),IRM(3,5),IRM(4,4),IRM(4,5),IRM(5,4),IRM(5,5)];
Fan_Low = [IRM(1,1),IRM(1,2),IRM(2,1)];
Fan_Medium = [IRM(1,3),IRM(2,2),IRM(2,3),IRM(3,2),IRM(3,3),IRM(4,3)];
Fan_High = [IRM(3,1),IRM(4,1),IRM(4,2),IRM(5,1),IRM(5,2),IRM(5,3)];

%------------------------------
% Remove Zero Elements
%

Fan_Off(Fan_Off==0) = [];
Fan_Low(Fan_Low==0) = [];
Fan_Medium(Fan_Medium==0) = [];
Fan_High(Fan_High==0) = [];

%------------------------------
% Calculate Areas of the Fuzzy Sets
%

b = 1;

Centr2 = [0 1 2.5 3];
a11 = [-0.5*Fan_Off+1];
a22 = [-0.5*Fan_Low+1];
a33 = [-0.5*Fan_Medium+1];
a44 = [-0.5*Fan_High+1];

Area11 = 0.5.*(a11+b).*Fan_Off;
Area22 = 0.5.*(a22+b).*Fan_Low;
Area33 = 0.5.*(a33+b).*Fan_Medium;
Area44 = 0.5.*(a44+b).*Fan_High;

%------------------------------
% center of Sums Method
%

CrispFanOutput = sum([Area11*Centr2(1),Area22*Centr2(2),Area33*Centr2(3),Area44*Centr2(4)])/...
    sum([Area11,Area22,Area33,Area44]);
end