function Y = prox_inf(X, lambda)
    % Get the size of the matrix
    [m, n] = size(X);
    
    % Initialize the output matrix Y
    Y = zeros(m, n);
    
    % Loop through each row of the matrix
    for i = 1:m
        % Extract the i-th row
        row_x = X(i, :);
        
        % Compute the L1 norm of the row
        row_norm1 = norm(row_x, 1);
        
        % Compute the proximal operator for the L_inf norm
        if row_norm1 > lambda
            Y(i, :) = row_x * (lambda / row_norm1);
        else
            Y(i, :) = row_x; % Leave the row unchanged
        end
    end
end
