function [y] = B3(x)

y = zeros (size (x));

i1 = find (-2 <= x & x < -1);
i2 = find (-1 <= x & x <= 1);
i3 = find (1 <= x & x <= 2);

y(i1) = .5 * (x(i1) + 2) .^ 3;
y(i2) = .5 * (3 * abs (x(i2)) .^ 3 - 6 * x(i2) .^ 2 + 4);
y(i3) = .5 * (2 - x(i3)).^3;
y = y/3;
end
