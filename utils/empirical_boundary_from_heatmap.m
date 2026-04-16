
function [rho_curve, alpha_curve] = empirical_boundary_from_heatmap(H, rho_vec, alpha_vec, thr)
% Extract an empirical boundary: for each rho, find the smallest alpha
% such that MSE <= thr. If none, set alpha = max(alpha_vec).
% H is (length(alpha_vec) x length(rho_vec)) with axis xy.

    A_len = numel(alpha_vec);
    R_len = numel(rho_vec);
    alpha_curve = zeros(1, R_len);

    for j = 1:R_len
        col = H(:, j);                             % MSE over alpha at fixed rho
        idx = find(col <= thr, 1, 'first');        % first "successful" alpha
        if isempty(idx)
            alpha_curve(j) = alpha_vec(end);
        else
            alpha_curve(j) = alpha_vec(idx);
        end
    end

    % light smoothing to avoid jagged line
    alpha_curve = local_smooth(alpha_curve, 7);

    rho_curve = rho_vec;
end
