function IzL1=Iono(A1,A2,A3,A4,t
%generates Ionospheric zenith delay on L1

if abs(t-A3)<(A4/4)
    IzL1c=A1+A2*cos((2*pi*(t-A3))/A4);
else
    IzL1c=A1;
end
end