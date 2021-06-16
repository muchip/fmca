clear all;
close all;
SampletBasis;
Points;
Indices;
Idcs = Idcs + 1;
sP = P(Idcs);

fun = exp(-20 * abs(sP-0.5)) +2*exp(-20 * abs(sP)) -exp(-20 * abs(sP+0.5));
fun = zeros(size(sP));
fun = randn(size(sP)); %(find(sP > -pi/10 & sP < pi/20)) = 1;
Tfun = T*fun';

axmax = max(max(T));
axmin = min(min(T));
aymin = min(P);
aymax = max(P);

figure(4)
for i = 1:4000
figure(4)
clf;
subplot(2,1,1)
plot(sP, fun,'b-');
hold on;
plot(sP, T(1:i,:)' * Tfun(1:i),'r.')
subplot(2,1,2)
plot(sP, T(i,:)','k.','linewidth',2)
axis([axmin axmax -0.5 0.5]);
pause(0.01)
end
figure(2);


% for i = 1:50
%     figure(2)
%     plot(P(Idcs),T(i,:),'b.','linewidth',2)
%     axis([axmin axmax aymin aymax]);
%     title(sprintf('samplet number %d', i));
%     pause(1)
% end