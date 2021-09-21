function [b3, db3, d2b3] = cbspline
b3 = @(x) fb3(x);
db3 = @(x) fdb3(x);
d2b3 = @(x) fd2b3(x);
end

function [y] = fb3(x)

y = zeros (size (x));

i1 = find (-2 < x & x < -1);
i2 = find (-1 <= x & x < 1);
i3 = find (1 <= x & x < 2);

y(i1) = .25 * (x(i1) + 2) .^ 3;
y(i2) = .25 * (3 * abs (x(i2)) .^ 3 - 6 * x(i2) .^ 2 + 4);
y(i3) = .25 * (2 - x(i3)).^3;

end

function [y] = fdb3(x)

y = zeros (size (x));

i1 = find (-2 < x & x < -1);
i2 = find (-1 <= x & x < 1);
i3 = find (1 <= x & x < 2);

y(i1) = .75 * (x(i1) + 2) .^ 2;
y(i2) = .25 * (9 * abs (x(i2)) .* x(i2) - 12 * x(i2));
y(i3) = -.75 * (2 - x(i3)) .^ 2;

end

function [y] = fd2b3(x)

y = zeros (size (x));

i1 = find (-2 < x & x < -1);
i2 = find (-1 <= x & x < 1);
i3 = find (1 <= x & x < 2);

y(i1) = 1.5 * (x(i1) + 2);
y(i2) = .25 * (18 * abs (x(i2)) - 12);
y(i3) = 1.5 * (2 - x(i3));

end