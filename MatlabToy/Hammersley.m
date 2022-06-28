function H = Hammersley(P,N)
%% 
% Author: Mansour Torabi
% Email:  smtoraabi@ymail.com
%%
% P: Number of samples in each dimensions
% N: Number of dimensions
% Hammersley Sequence:
% Ref: Wong, Tien-Tsin, Wai-Shing Luk, and Pheng-Ann Heng. "Sampling with 
% Hammersley and Halton points." Journal of graphics tools - 1997
H(1,:)=(0:P-1)/P;
PP = primes(1000);
for i = 2:N  
  for j = 0:P-1
      J = j;
      H(i,j+1)=0;
      C=1/PP(i-1);
      while J > 0
          R0 = rem(J,PP(i-1));
          H(i,j+1) = H(i,j+1) + R0*C;
          
          C = C / PP(i-1);
          J = floor(J/PP(i-1));
      end
  end
end