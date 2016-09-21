function [xyzEst,clockBiasEst,PDOP,TDOP,GDOP]=LLSPos(Satxyz,pr,nomXYZ,clockBiasNom,Nsats,c,W,trop,ion)
%Linear Least Squares positioning
%Sean Lantto

%pre-allocation
% prComp=zeros(nSat,1);
% uNom2Sat=zeros(nSat,3);

for k=1:Nsats
    %find computed pseudorange for each sat in epoch i
    prComp(k)=norm(Satxyz(k,:)-nomXYZ)+clockBiasNom*c;
    uNom2Sat(k,:)=(Satxyz(k,:)-nomXYZ)/norm(Satxyz(k,:)-nomXYZ);
    
    %Atmospheric delays
    if (trop)
        prCompT(k)=prComp(k)+mel*TropSaastamoinen(llh,Po,To,eo);
    end
    if (ion)
        
    end
end

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




end
