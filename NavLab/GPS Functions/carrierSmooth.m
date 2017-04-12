function prSmooth=carrierSmooth(PhaseL1,PhaseL2,M,prDataIF)

phaseIF=(2.546*PhaseL1)-(1.546*PhaseL2);
phaseIF=phaseIF';
prSmooth(1,:)=prDataIF(1,:);

for j=2:length(prDataIF)
    prSmooth(j,:)=((1/M)*prDataIF(j,:))'+(((M-1)/M)*(phaseIF(:,j)-phaseIF(:,j-1)+prSmooth(j-1,:)));
end
end