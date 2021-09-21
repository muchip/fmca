function [G, N, detT, T, D2G] = Gamma (fr, fDr, fD2r, x)

% [G, N, detT, T, D2T] = Gamma (fr, fDr, fD2r, x)
% returns the data of the boundary with...
% G is a two column vector containing the boundary points in (x,y) format
% N is the unnormed normal vector at each point in G in (x,y) format
% detT is the norm of the tangent vector at each point in G.
% T is the tangent vector at each point in G in  (x,y) format
% D2G is the second derivative of the boundary in (x,y) format
% ... for a given inputvector x

% ensure, that x is a column vector

if (1 == min (size (x)))
    
    x = reshape (x, length (x), 1);

% init points (X), sinX (s), cosX (c), radius (r)
    s    = zeros (size (x));
    c    = zeros (size (x));
    r    = zeros (size (x));
    Dr   = zeros (size (x));
    D2r  = zeros (size (x));
    detT = zeros (size (x));

    n    = length (x);
    G    = zeros (n,2);
    N    = zeros (n,2);
    D2G  = zeros (n,2);
    T    = zeros (n,2);

    s = sin (2 * pi * x);
    c = cos (2 * pi * x);

% evaluate derivatives of the radius
    r   = fr (x);
    Dr  = fDr (x);
    D2r = fD2r (x);

% compute boundary information outer normal (N), volume (detT), second ...
% derivative of the boundary 
    G    = [r .* c, r .* s];
    T    = [Dr .* c - 2 * pi * r .* s, Dr .* s + 2 * pi * r .* c];
    detT = sqrt (Dr .^ 2 + (2 * pi * r).^2);
    N    = [Dr .* s + 2 * pi * r .* c, 2 * pi * r .* s - Dr .* c];
    D2G  = [(D2r - (2 * pi * 2 * pi) * r) .* c - 4 * pi * Dr .* s, ...
           (D2r - (2 * pi * 2 * pi) * r) .* s + 4 * pi * Dr .* c];
  
else
    
    disp ('input is not a vector, try help')

end

end