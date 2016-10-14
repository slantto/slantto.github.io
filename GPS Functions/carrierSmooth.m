function prSmooth=carrierSmooth(phaseL1,phaseL2,M,prDataIF)

phaseIF=(2.546*phaseL1)-(1.546*phaseL2);
prSmooth(1)=prDataIF(1);

for j=2:length(prDataIF)
    prSmooth(j)=((1/M)*prDataIF(j))+(((M-1)/M)*(phaseIF(j)-phaseIF(j-1)+prDataIF(j-1)));
end
end