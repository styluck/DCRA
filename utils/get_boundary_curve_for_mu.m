function [rho_curve, alpha_curve, src] = get_boundary_curve_for_mu(mu, rho_vec, alpha_vec)
% Try to load a theoretical boundary curve for this mu.
% File format (if available): boundary_mu_<value>.mat containing
%   rho_bdry : vector of rho values in [0,1]
%   alpha_bdry : same-sized vector of alpha values in [0,1]
%
% Return empty if not found; caller will use empirical fallback.

    rho_curve = [];
    alpha_curve = [];
    src = '';

    % build candidate filenames (plain and pretty mu)
    mu_strs = {num2str(mu), sprintf('%.0f',mu), sprintf('%.1f',mu), sprintf('%.2f',mu)};
    for k = 1:numel(mu_strs)
        fname = sprintf('boundary_mu_%s.mat', mu_strs{k});
        if exist(fname,'file') == 2
            S = load(fname);
            if isfield(S,'rho_bdry') && isfield(S,'alpha_bdry')
                rho_curve = S.rho_bdry(:)';
                alpha_curve = S.alpha_bdry(:)';
                % Clip to plotting range and sort by rho
                mask = rho_curve >= min(rho_vec) & rho_curve <= max(rho_vec);
                rho_curve = rho_curve(mask);
                alpha_curve = alpha_curve(mask);
                [rho_curve, I] = sort(rho_curve);
                alpha_curve = alpha_curve(I);
                % Smooth slightly (optional)
                alpha_curve = local_smooth(alpha_curve, 5);
                src = 'theory';
                return;
            end
        end
    end
end

