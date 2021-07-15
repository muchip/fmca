clear all;
close all;
SampletBasis;
Points;
Indices;
Idcs = Idcs + 1;
sP = P(Idcs);
sortP = sort(P);
fun = 1.5 * exp(-40 * abs(sP-0.25)) +2*exp(-40 * abs(sP)) -exp(-40 * abs(sP+0.5));
%fun = zeros(size(sP));
%fun = randn(size(sP)); %(find(sP > -pi/10 & sP < pi/20)) = 1;
%fun = zeros(size(sP));
%fun(1) = randn(1);
%for i = 2:length(fun)
%    fun(i) = fun(i-1) + sqrt(sP(i)-sP(i-1)) * randn(1);
%end
Tfun = T*fun';
mCoeff = max(abs(Tfun));
Tfun3 = Tfun;
Tfun2 = Tfun;
Tfun1 = Tfun;
Tfun3(find(abs(Tfun) < 1e-3 * mCoeff)) = 0;
Tfun2(find(abs(Tfun) < 1e-2 * mCoeff)) = 0;
Tfun1(find(abs(Tfun) < 1e-1 * mCoeff)) = 0;
1- nnz(Tfun3) / size(sP,2)
1- nnz(Tfun2) / size(sP,2)
1- nnz(Tfun1) / size(sP,2)

%BM compression
% ans =
% 
%     0.9842
% 
% 
% ans =
% 
%     0.9985
% 
% 
% ans =
% 
%     0.9997
%EXP compression
% ans =
% 
%     0.9855
% 
% 
% ans =
% 
%     0.9917
% 
% 
% ans =
% 
%     0.9963
% M = [sP', fun', Tfun, T' * Tfun1, T' * Tfun2, T' * Tfun3];
% save('BMCompress1D.txt','M','-ascii');
% 
plot(sP, fun,'b-');
hold on
plot(sP, T' * Tfun,'r.')

axmax = max(max(T));
axmin = min(min(T));
aymin = min(P);
aymax = max(P);

figure(4)
for i = 1:4000
%figure(4)
clf;
%subplot(2,1,1)
%plot(sP, fun,'b-');
%hold on;
%plot(sP, T(1:i,:)' * Tfun(1:i),'r.')
%subplot(2,1,2)
plot(sP, T(i,:)','k.','linewidth',2)
axis([axmin axmax -0.5 0.5]);
if(mod(i,20) == 1)
pause()
end
end
figure(2);


for i = 1:50
    figure(2)
    plot(P(Idcs),T(i,:),'b.','linewidth',2)
    axis([axmin axmax aymin aymax]);
    title(sprintf('samplet number %d', i));
    pause(1)
end