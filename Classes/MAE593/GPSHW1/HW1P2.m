%HW1P2
%Sean Lantto
clear all

load mysteryCA
%add noise to mysteryCA
N=rand(1);
mysteryCA(mysteryCA==0)=-1;
mysteryCA=mysteryCA+N*randn(1,1023);

%ID the mysteryCA
for i=1:37
    prn=cacode(i);
    prn(prn==0)=-1;
    caID=xcorr(mysteryCA(1:end-9),prn(10:end),'coeff');
    %found this upper limit by cross correlating different PRNs and
    %knowning that only an autocorrelation would produce a large spike
    %could also xcorr prn(i) with itself and then prn(i) with mysteryCA
    %and if they are equal than the loop breaks
    if max(caID)>0.6
        break
    else
        continue
    end
end

fprintf('The mystery CA is PRN %d .\n',i)

%MysteryCA found to be PRN 12, plots below show cross and auto correlation
%to prove this. Function is unable to identify CA code as N aprroaches 1.3,
%but is still able to id it at 1.2999

subplot(511),stairs(mysteryCA(1:100))
title('C/A code identification')
legend('Mystery C/A Code')
ylim([-2 2])

subplot(512),stairs(prn(1:100))
legend('C/A Code PRN12')
ylim([-2 2])

prnim=cacode(i-1);
prnim(prnim==0)=-1;
subplot(513),stairs(prnim(1:100))
legend('C/A Code PRN11')
ylim([-2 2])


subplot(514),plot(xcorr(mysteryCA(1:end-9),prn(10:end),'coeff'),'r--','linewidth',2)
hold on
plot(xcorr(mysteryCA(1:end-9),prnim(10:end),'coeff'),'linewidth',2)
ylim([-1 1])
xlim([1023-100,1023+100])
legend('xcorr(mysteryCA,PRN12)','xcorr(mysteryCA,PRN11)')
hold off