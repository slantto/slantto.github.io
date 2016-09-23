function [xyzEst,clockBiasEst,PDOP,TDOP,GDOP,preFit,postFit]=LLSPos(Satxyz,pr,nomXYZ,clockBiasNom,Nsats,c,W,trop,ion,mel,llh,Po,To,eo,A1,A2,A3,A4,t,el,klo,tshell,ionfree)
%Linear Least Squares positioning
%Sean Lantto

%pre-allocation
% prComp=zeros(nSat(i));
% uNom2Sat=zeros(nSat(i),3);

for k=1:Nsats
    %find computed pseudorange for each sat in epoch i
    prComp(k)=norm(Satxyz(k,:)-nomXYZ)+clockBiasNom*c;
    uNom2Sat(k,:)=(Satxyz(k,:)-nomXYZ)/norm(Satxyz(k,:)-nomXYZ);
    if trop==1
        prComp(k)=prComp(k)+mel(k)*TropSaastamoinen(llh,Po,To,eo);
    end
    if ion==1
        if klo==1
        OFk=1+16*(0.54-(el(k)/180));
        prComp(k)=prComp(k)+OFk*Iono(A1,A2,A3,A4,t,c);
        
        elseif tshell==1
        
        elseif ionfree==1
        
        end
    end
end


%Estimate position without atmosphere
G=horzcat(-uNom2Sat,ones(Nsats,1)); %form geometery matrix for each epoch
H=inv(G'*W*G);

%find deltaRho and Delta X
deltaRho=pr-prComp';
dX=H*G'*W*deltaRho;
%estimate position and clock bias
xyzEst=nomXYZ'+dX(1:3);
clockBiasEst=clockBiasNom+(dX(4)/c);

PDOP=sqrt(H(1,1)+H(2,2)+H(3,3));
TDOP=sqrt(H(4,4));
GDOP=trace(H);
preFit=deltaRho;
postFit=pinv(H')*H'*H*dX;


end
