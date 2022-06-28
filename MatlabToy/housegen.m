function [u,nu] = housegen(x)
    % [u,nu] = housegen(x)
    % Generate Householder reflection.
    % G. W. Stewart, "Matrix Algorithms, Volume 1", SIAM, 1998.
    % [u,nu] = housegen(x).
    % H = I - uu' with Hx = -+ nu e_1
    %    returns nu = norm(x).
    u = x;
    nu = norm(x);
    if nu == 0
        u(1) = sqrt(2);
        return
    end
    u = x/nu;
    if u(1) >= 0
        u(1) = u(1) + 1;
        nu = -nu;
    else
        u(1) = u(1) - 1;
    end
    u = u/sqrt(abs(u(1)));
end