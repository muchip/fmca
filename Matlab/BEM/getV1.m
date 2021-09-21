function V1 = getV1 (n)

% V1 = getV1(n)
% computes the singular part of the single layer operator
% analytically by using the fast fourier transform to evaluate the
% formula which can be found in Kress: linear integral equations
% this formula only works, if n is an equal number

if mod (n,2)
    disp ('n is odd, you are screwed! try help...')
    
else
    % length of the trigonometric polynomial
    m = n / 2;

    % get memory for V1 and a
    V1 = zeros (n);
    a = zeros (1, m+1);

    % compute real fourier coefficients
    a(2:m+1) = 1 ./ (1:m);
    a(m+1) = a(m+1) * 0.5;

    % compute the conjugate coefficients
    y = fliplr (a(2:m));

    % compute complex coefficients and mess up weights (.5)
    c = [a,y];

    % compute inverse fft and mess up weigths from the original
    % fourier series and the coefficients c which is -1/m * .5 = -1/n
    % and the fucking weight of the ifft function which is 1/n
    f = ifft (-c);

    % plug everything in a toeplitz matrix
    V1 = toeplitz (f);
end