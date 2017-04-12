clear all

%% Problem 1
%Mass Matrix
M=[2,0,0;0,3,1;0,1,1];
%Stiffness Matrix
K=[2,-1,0;-1,1,0;0,0,2];

%Solve for modal vector and e-values
[phi,d]=eig(K,M);
%Sort natural frequencies and modes
[omega,id]=sort(sqrt(diag(d)));

phi=phi(:,id);

alpha1=sqrt(phi(:,1)'*M*phi(:,1));
alpha2=sqrt(phi(:,2)'*M*phi(:,2));
alpha3=sqrt(phi(:,3)'*M*phi(:,3));

%normalize modes
phinorm1=phi(:,1)/alpha1;
phinorm2=phi(:,2)/alpha2;
phinorm3=phi(:,3)/alpha3;

%enforce initial conditions
x0=[1;1;.5];
xdot0=[0;1;0];

c1=phinorm1'*M*x0;
c2=phinorm2'*M*x0;
c3=phinorm3'*M*x0;

s=xdot0./omega;

%% Problem 2

M2=[6,1,1;1,1,0;1,0,1];
K2=[1,0,0;0,6,0;0,0,6];

CharEq=[-4 0 61 0 -228 0 36];
mu=roots(CharEq);

