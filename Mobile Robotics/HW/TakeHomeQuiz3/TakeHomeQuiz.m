%Take Home Quiz 3
%Sean Lantto
%March 22, 2017


%Problem 1
PQR=horzcat(SD.P',SD.Q',SD.R');
covPQR=cov(PQR);

figure;hist(SD.P); 
figure;hist(SD.Q);
figure;hist(SD.R);


%Problem 2


%Problem 3
mu=[2;0];
sigma=[1,0.9;0.9,1];
for i=1:500
X(:,i)=mvnrnd(mu,sigma)';
Y(:,i)=[2,-1;3,-5]*X(:,i);
Z(:,i)=[sin(X(1,i))+X(2,i)^2;X(1,i)*Y(2,i)];
end

figure;scatter(X(1,:),X(2,:))
figure;hist(X');
figure;scatter(Y(1,:),Y(2,:))
figure;hist(Y');
figure;scatter(Z(1,:),Z(2,:))
figure;hist(Z');

%Problem 4
M=1; %mass (kg)
fv=1; %damping coefficient (N-s/m)
K=5; %spring constant (N/m)

sigmax=0.2; %std dev of pos sensor
sigmaf=0.05; %std dev of force sensor