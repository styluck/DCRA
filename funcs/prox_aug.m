function [u_prox, Act_set, Inact_set] = prox_aug(u, gamma, c_z, M)
% PROX_AUG  Proximal operator of f(u) = ||u(1:M)||_1 + c_z' * u(M+1:end)
%
% Input:
%   u     : input vector of length M+N
%   gamma : proximal parameter (>0)
%   c_z   : vector of length N (the linear coefficients)
%   M     : size of the first block (l1 part)
%
% Output:
%   u_prox   : proximal point
%   Act_set  : active set indices for the l1-part
%   Inact_set: inactive set indices for the l1-part

    % Split blocks
    u1 = u(1:M);
    u2 = u(M+1:end);

    %% Block 1: l1 norm
    a = abs(u1) - gamma;
    Act_set = (a > 0);
    u1_prox = (Act_set .* sign(u1)) .* a;

    if nargout == 3
        Inact_set = (a <= 0);
    end

    %% Block 2: linear term <c_z, u2>
    % prox_{gamma <c_z,·>} (u2) = u2 - gamma * c_z
    u2_prox = u2 - gamma * c_z;

    %% Combine
    u_prox = [u1_prox; u2_prox];
end
