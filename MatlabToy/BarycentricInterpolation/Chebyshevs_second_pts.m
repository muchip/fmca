%% Chebyshevs' points "second type"
function LP = Chebyshevs_second_pts(n)
LP = cos(pi*(0:n-1)/(n-1));
LP = .5 * LP + .5;
end
