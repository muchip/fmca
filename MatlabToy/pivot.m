function R = pivot(M, r, c)
    [d, w] = size(M); % Get matrix dimensions
    R = zeros(d, w); % Initialize to appropriate size
    R(:,c) = M(:, c) / M(r,c); % Copy row r, normalizing M(r,c) to 1
    for k = 1:w % For all matrix rows
        if (k ~= c) % Other then r
            R(:,k) = M(:,k) ... % Set them equal to the original matrix
            - M(r,k) * R(:,c); % Minus a multiple of normalized row r, making R(k,c)=0
        end
    end
end