function [rho_grid, alpha_c] = rs_boundary_curve(mu, rho_grid, alpha_grid, Nmc)
% RS boundary for given mu by scanning alpha for each rho.
% Nmc = number of Monte Carlo samples for the Gaussian average (e.g., 2e4).

alpha_c = nan(size(rho_grid));
for ir = 1:numel(rho_grid)
    rho = rho_grid(ir);
    % warm-start guesses
    p = rho; m = 0.5*rho; Q = rho; q = p^2;

    found = false;
    for ia = 1:numel(alpha_grid)
        alpha = alpha_grid(ia);

        % Fixed-point iterations
        for it = 1:500
            Delta = max(Q - q, 1e-12);

            % Conjugates (Eqs. 19–22)
            tQ = alpha/Delta;
            tm = alpha/Delta;
            ttau = alpha*((rho - 2*m + Q) + mu^2*(rho - p)^2)/Delta^2;
            tp = alpha*mu^2*(rho - p)/Delta;

            % Monte Carlo averages for p,m,Q,q via x* = clip(soft(h/tQ, 1/tQ))
            t = randn(Nmc,1);
            x0 = double(rand(Nmc,1) < rho);
            h = tm*x0 + sqrt(ttau)*t + tp;
            u = h ./ tQ;
            xsoft = sign(u).*max(abs(u) - 1/tQ, 0);
            xstar = min(max(xsoft, 0), 1);   % clip to [0,1]

            p_new = mean(xstar);
            m_new = mean(x0 .* xstar);
            Q_new = mean(xstar.^2);
            q_new = p_new.^2;                % RS closure

            % check convergence
            vec_old = [p;m;Q;q];
            vec_new = [p_new;m_new;Q_new;q_new];
            if norm(vec_new - vec_old, inf) < 1e-7, p=p_new; m=m_new; Q=Q_new; q=q_new; break; end
            p=p_new; m=m_new; Q=Q_new; q=q_new;
        end

        MSE = Q - 2*m + rho;
        if MSE <= 1e-6
            alpha_c(ir) = alpha; found = true; break;
        end
    end
    if ~found, alpha_c(ir) = alpha_grid(end); end
end
end
