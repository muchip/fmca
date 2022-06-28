%% Chebyshevs points "first kind"
function LP = Chebyshevs_first_pts(n)
LP = cos(((2 * (0:n-1) + 1) * pi)/(2 * (n-1) + 2)); % Chebychevs' points
LP = .5 * LP + .5;
end
