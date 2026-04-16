function [rho_curve, alpha_curve] = rs_boundary_curve1(mu, rho_vec, alpha_vec, Nmc, tol, maxit)
% RS_BOUNDARY_CURVE  Replica-symmetric reconstruction limit alpha_c(rho)
%   for binary CS with A_ij ~ N(mu/N, 1/N) and x0 ∈ {0,1}.
%
% Inputs:
%   mu         : bias parameter (scalar), e.g., 0, 2, 10
%   rho_vec    : 1×R vector of densities rho in (0,1]
%   alpha_vec  : 1×A vector of compression ratios alpha in (0,1]
%   Nmc        : # Monte Carlo samples for Gaussian average (e.g., 2e4)
%   tol        : fixed-point tolerance (e.g., 1e-7)
%   maxit      : max fixed-point iters per (alpha,rho) (e.g., 500)
%
% Outputs:
%   rho_curve   : = rho_vec (returned for convenience)
%   alpha_curve : same length as rho_vec; alpha_c(rho) at RS threshold
%
% Notes:
%   - Uses zero-temperature RS fixed-point with single-site minimization
%     x* = clip_{[0,1]}( soft( (h)/tQ, 1/tQ ) ), where
%       h = tM*x0 + sqrt(tTau)*t + tP,
%       (tQ, tM, tTau, tP) are the conjugates.
%   - Success is declared when RS-MSE = Q - 2m + rho <= 1e-6.
%
% Reference:
%   See main text for equations mapping (RS conjugates & moments).

    if nargin < 4 || isempty(Nmc),  Nmc  = 20000; end
    if nargin < 5 || isempty(tol),  tol  = 1e-7;  end
    if nargin < 6 || isempty(maxit),maxit= 500;   end

    rho_vec   = rho_vec(:)';   % row
    alpha_vec = alpha_vec(:)'; % row

    alpha_curve = nan(size(rho_vec));

    % Pre-generate Gaussian noise to reduce variance (reused per alpha)
    t_gauss = randn(Nmc,1);

    for ir = 1:numel(rho_vec)
        rho = rho_vec(ir);

        % warm-start order parameters
        p = max(1e-12, rho);
        m = 0.5 * rho;
        Q = max(1e-12, rho);
        q = p^2;

        found = false;

        for ia = 1:numel(alpha_vec)
            alpha = alpha_vec(ia);

            % fixed-point iteration on (p,m,Q,q)
            for it = 1:maxit
                Delta = max(Q - q, 1e-12);

                % Conjugate parameters (compact RS forms)
                tQ   = alpha / Delta;
                tM   = alpha / Delta;
                tTau = alpha * ( (rho - 2*m + Q) + mu^2 * (rho - p)^2 ) / (Delta^2);
                tP   = alpha * mu^2 * (rho - p) / Delta;

                % Monte Carlo mixture over x0 ~ Bernoulli(rho), t ~ N(0,1)
                x0_mc = double(rand(Nmc,1) < rho);
                h     = tM .* x0_mc + sqrt(max(tTau,0)) .* t_gauss + tP;

                % single-site minimizer: soft then clip to [0,1]
                u     = h ./ max(tQ, 1e-12);
                xsoft = sign(u) .* max(abs(u) - 1./max(tQ,1e-12), 0);
                xstar = min(max(xsoft, 0), 1);

                % RS moment updates
                p_new = mean(xstar);
                m_new = mean(x0_mc .* xstar);
                Q_new = mean(xstar.^2);
                q_new = p_new.^2; % RS closure

                if max(abs([p_new-p, m_new-m, Q_new-Q, q_new-q])) < tol
                    p = p_new; m = m_new; Q = Q_new; q = q_new;
                    break;
                end
                p = p_new; m = m_new; Q = Q_new; q = q_new;
            end

            % RS-predicted MSE
            MSE = Q - 2*m + rho;

            if MSE <= 1e-6
                alpha_curve(ir) = alpha;
                found = true;
                break;
            end
        end

        if ~found
            alpha_curve(ir) = alpha_vec(end);
        end
    end

    rho_curve = rho_vec;
end
