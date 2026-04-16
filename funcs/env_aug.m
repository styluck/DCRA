function [h, dh] = env_aug(x, del, OPTS)
% ENV_AUG  Moreau envelope of f(u) = ||u(1:M)||_1 + c_z' * u(M+1:end)
%
% Input:
%   x   : input vector of length M+N
%   del : smoothing parameter (>0)
%   c_z : vector of length N (linear coefficients)
%   M   : size of the first block (l1 part)
%
% Output:
%   h   : envelope value at x
%   dh  : (optional) gradient at x

    % Compute prox
    c_z = OPTS.c_z;
    M = OPTS.M;
    u_prox = prox_aug(x, del, c_z, M);

    % Evaluate envelope: f(u_prox) + (1/(2*del))*||x - u_prox||^2
    %   f(u_prox) = ||u1||_1 + c_z' * u2
    u1 = u_prox(1:M);
    u2 = u_prox(M+1:end);

    fval = sum(abs(u1)) + sum(c_z .* u2);
    h = fval + 0.5 * (norm(x - u_prox, 'fro')^2) / del;

    % Gradient if requested
    if nargout > 1
        dh = (x - u_prox) ./ del;
    end
end
