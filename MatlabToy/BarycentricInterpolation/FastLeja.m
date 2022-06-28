function zt = FastLeja(n)
a = 0; b = 1; nflp = n;
if abs(a) > abs(b), zt = [a,b]; else zt = [b,a]; end
zt(3) = (a+b)/2;
zs(1) = (zt(2)+zt(3))/2; zs(2) = (zt(3)+zt(1))/2;
zprod(1) = prod(zs(1)-zt); zprod(2) = prod(zs(2)-zt);
index(1,1) = 2; index(1,2) = 3; index(2,1) = 3; index(2,2) = 1;
for i = 4:nflp
[~,maxi] = max(abs(zprod));
zt(i) = zs(maxi);
index(i-1,1) = i; index(i-1,2) = index(maxi,2); index(maxi,2) = i; zs(maxi) = (zt(index(maxi,1))+zt(index(maxi,2)))/2;
zs(i-1) = (zt(index(i-1,1))+zt(index(i-1,2)))/2;
zprod(maxi) = prod(zs(maxi)-zt(1:i-1));
zprod(i-1) = prod(zs(i-1)-zt(1:i-1));
zprod = zprod.*(zs-zt(i));
end
end