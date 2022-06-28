function [U,R] = hqrd(X)  
    % Householder triangularization.  [U,R] = hqrd(X);
    % Generators of Householder reflections stored in U.
    % H_k = I - U(:,k)*U(:,k)'.
    % prod(H_m ... H_1)X = [R; 0]
    % where m = min(size(X))
    % G. W. Stewart, "Matrix Algorithms, Volume 1", SIAM, 1998.
    [n,p] = size(X);
    U = zeros(size(X));
    m = min(n,p);
    R = zeros(m,m);
    for k = 1:min(n,p)
        [U(k:n,k),R(k,k)] = housegen(X(k:n,k));
        v = U(k:n,k)'*X(k:n,k+1:p);
        X(k:n,k+1:p) = X(k:n,k+1:p) - U(k:n,k)*v;
        R(k,k+1:p) = X(k,k+1:p);
    end
end