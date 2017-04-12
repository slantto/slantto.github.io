for i=2:1:length(SD.R)
    Angle_Intb(i)=-(SD.R(i)-8.464846872172502e-04)*SD.Time_Diff(i)+Angle_Int(i-1);
end