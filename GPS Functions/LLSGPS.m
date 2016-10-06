function [xyzEst,clockBiasEst,preFit,postFit,el]= LLSGPS(nomXYZ,clockBiasNom,satXYZ,prObserved, nSat,weight, tod,iono,trop)

%Inputs from data sets to lecture 1 format
%XYZTruth=truthXYZ;
%clockBiasTruth=truthClockBias;
SOL=299792458;

%Creating loop for the number of epochs
%Creating zeros matrix to fill in computed postitions
prComputed=zeros(nSat,1);
uNom2Sat=zeros(nSat,3);
%Creating loop for changes in number of Sats
for j=1:nSat
    prComputed(j)=norm(satXYZ(j,:)-nomXYZ)+clockBiasNom*SOL;
     if (iono)
         disp(OFIONO(satXYZ(j,:),nomXYZ)*ionoKlobuchar(7.002209958205558e+04+mod(tod,86400)))
         prComputed(j)=prComputed(j)+OFIONO(satXYZ(j,:),nomXYZ)*(ionoKlobuchar(7.002209958205558e+04+mod(tod,86400)));
     end
     if(trop)
          prComputed(j)=prComputed(j)+tropMEL(satXYZ(j,:),nomXYZ)*tropZSaastemoinen(Po,e0,Ho,Lat,Lon);
     end
    uNom2Sat(j,:)=(satXYZ(j,:)-nomXYZ)/norm(satXYZ(j,:)-nomXYZ);
%     [el(j),A]=calcElAz(nomXYZ,satXYZ(j,:));
end
if (weight==1)
    W=zeros(nSat);
    for j=1:nSat
        W(j,j)=sin(el(j));
    end
else
    
    W=eye(nSat);
end
%Creating H matrix
H=horzcat(uNom2Sat,ones(nSat,1));
%Solving for xyz and clock bias estimations
deltaRho=prComputed-prObserved;
preFit=deltaRho;
dX=inv(H'*W*H)*H'*W*deltaRho;


postFit=pinv(H')*H'*H*dX;
xyzEst=nomXYZ+dX(1:3)';

clockBiasEst=clockBiasNom+-1*dX(4)/SOL;






