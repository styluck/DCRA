function [xsol, outs] = DCRA_intpl(A, b, V, env, OPTIONS)
%% ***************************************************************
% filename: QAfactor
% to solve the factorized form of the penalty problem of (3) induced by
% the DC constraint
% \underset{x\in \{-1, 1\}^n}{\min} \|Ax-b\|_1 
%% ***************************************************************

if nargin < 5 || isempty(OPTIONS)
    OPTIONS = struct();
end

m = size(A, 2);
n = m + 1;
if nargin < 3 || isempty(V)
    V = randn(5, n);
end

% prox = @prox_l1;
V = retr_ob(V); % generate the initial point V\in Ob(n,r)

sn = sqrt(n);
if isfield(OPTIONS,'gaptol');    gap_tol  = OPTIONS.gaptol;    else; gap_tol =1e-6; end % 1e-8;
if isfield(OPTIONS,'objtol');    obj_tol  = OPTIONS.objtol;    else; obj_tol =1e-6; end % 1e-8;
if isfield(OPTIONS,'xtol');      xtol     = OPTIONS.xtol;      else; xtol =1e-4*sn; end
if isfield(OPTIONS,'ftol');      ftol     = OPTIONS.ftol;      else; ftol =1e-4*sn; end
if isfield(OPTIONS,'maxiter');   maxiter  = OPTIONS.maxiter;   else; maxiter =1e3;  end
if isfield(OPTIONS,'submxitr');  submxitr = OPTIONS.submxitr;  else; submxitr =5;  end
if isfield(OPTIONS,'printyes');  printyes = OPTIONS.printyes;  else; printyes =0;   end
if isfield(OPTIONS,'recordALM'); recordALM = OPTIONS.recordALM;else; recordALM =0;  end
% initial penalty param
if isfield(OPTIONS,'rho');       rho = OPTIONS.rho;            else; rho = 1;       end
% envelope smooth param
if isfield(OPTIONS,'del');       del = OPTIONS.del;            else; del =5;     end

% ***************** The factor_penalty method *****************************
if (printyes)
    fprintf('\n **********************************************************************************');
    fprintf('\n %-4s %-12s %-10s %-4s %-12s %-12s %-12s\n', ...
        'iter', 'rho', 'gapk', 'obj', 'subiter', 'rankX', 'time');

end

% ************************* Main Loop *********************************
tstart = tic;
sigma = 1.2; % increment of the penalty param 

[P, D, Q] = svd(V,'econ');
P1 = P(:,1);
Q1 = Q(:,1);

d = diag(D);

clear P Q D

Gmma = -2*(d(1)*P1)*Q1';
Lf = .5*abs(svds(A,1)); % norm(A,2);

outs.time = zeros(maxiter, 1);
outs.his = zeros(maxiter, 1);
outs.infeas = zeros(maxiter, 1);
% objold = inf;
for iter = 1:maxiter

    % solve the augmented lagrangian subproblem
    if (recordALM)
        fprintf('%-4s %-10s %-10s %-10s %-10s %-10s %-10s\n', ...
            'itr', 'subitr', 'g(x)', 'fval', 'dif_Lmb', 'beta', 'time');
        tstart = tic;
    end

    obj_list = zeros(submxitr, 1);

    % current outer iterate as starting point for inner loop
    Vold = V;

    % objective at outer iterate (for outer stopping test)
    vv  = V(:,2:end)'*V(:,1);
    AVb = A*vv - b;
    [objold, ~] = env(AVb, del, OPTIONS);   % we don't reuse dh here

    % FISTA parameters for extrapolation (reset per outer iteration)
    told = 1;
    t    = 1;

    for subitr = 1:submxitr
        %--------------------------------------------------------------
        % 1) FISTA-style extrapolation: Vkt = V + beta (V - Vold)
        %--------------------------------------------------------------
        beta = (told - 1)/t;
        % optional: cap beta in [0, beta_ex] if you want a fixed upper bound
        % beta = max(0, min(beta, beta_ex));

        Vkt = V + beta*(V - Vold);

        %--------------------------------------------------------------
        % 2) Envelope and gradient at extrapolated point Vkt
        %--------------------------------------------------------------
        vv_kt  = Vkt(:,2:end)'*Vkt(:,1);
        AVb_kt = A*vv_kt - b;
        [obj_kt, dh_kt] = env(AVb_kt, del, OPTIONS); %#ok<NASGU>

        % gradient of smoothed part at Vkt
        grad_kt = 2*Vkt*dual_A(dh_kt, A);

        %--------------------------------------------------------------
        % 3) MM step centered at Vkt: unconstrained step then projection
        %--------------------------------------------------------------
        SCALAR = 1/(2*rho + Lf);
        OUTER  = Lf*Vkt - rho*Gmma - grad_kt;
        Vnew   = retr_ob(Vkt, SCALAR*OUTER);

        %--------------------------------------------------------------
        % 4) Objective, SVD, DC subgradient, rank-one gap at Vnew
        %--------------------------------------------------------------
        vv   = Vnew(:,2:end)'*Vnew(:,1);
        AVb  = A*vv - b;
        [obj, ~] = env(AVb, del, OPTIONS);   % objective at new point

        [P, D, Q] = svd(Vnew, 'econ');
        P1 = P(:,1);
        Q1 = Q(:,1);
        d  = diag(D);
        d1 = d(1);
        Gmma_new = -2*(d1*P1)*Q1';

        gapk = sum(d.^2) - d1^2;   % ||V||_F^2 - ||V||_2^2

        gsubiter = 0;

        % recorder
        if (recordALM) && rem(subitr,5) == 0
            ttime    = toc(tstart);
            % normAVbW, bta assumed defined as in your original code
            fprintf('%-4d %-4d %-10.2e %-10.2e %-10.2e %-10.2e %-10.1f\n', ...
                subitr, gsubiter, gapk, obj, normAVbW, bta, ttime);
        end

        % penalized objective value at Vnew
        obj_list(subitr) = obj + rho*gapk;

        %--------------------------------------------------------------
        % 5) Inner stopping condition (same structure as before)
        %--------------------------------------------------------------
        if (subitr >= 10) && (  ...
                (abs(obj - obj_list(subitr - 9))/(1 + abs(obj)) <= ftol) || ...
                (norm(Vold - Vnew,'fro') < xtol)  )
            V = Vnew;
            Gmma = Gmma_new;
            % update FISTA t for consistency (not really needed after break,
            % but keeps the state coherent if you ever reuse t,told)
            told = t;
            t    = 0.5*(1+sqrt(1+4*told^2));
            break;
        end

        %--------------------------------------------------------------
        % 6) Prepare for next inner iteration: shift iterates and Gmma,
        %    update FISTA parameter t
        %--------------------------------------------------------------
        Vold = V;
        V    = Vnew;
        Gmma = Gmma_new;

        told = t;
        t    = 0.5*(1+sqrt(1+4*told^2));
    end

    % end of inner loop: V is the last inner iterate
    ttime = toc(tstart);
    outs.time(iter)   = ttime;
    outs.his(iter)    = obj;
    outs.infeas(iter) = gapk;

    if (printyes) && (mod(iter,5)==0)
        rankX = length(d(d>1e-10));
        fprintf(' %-4d %-10.2e %-10.2e %-10.2e %-10d %-10d %-10.2f\n', ...
            iter, rho, gapk, obj, subitr, rankX, ttime);
    end

    % stopping criterion for outer iterate (unchanged)
    if gapk < gap_tol && ...
       abs(objold - obj_list(subitr))/max(1,abs(obj_list(subitr))) < obj_tol
        break;
    else
        rho = min(sigma*rho,1.0e+6); % increase the penalty param
    end

    objold = obj_list(subitr);
end

if (printyes)

    ttime = toc(tstart);
    rankX= length(d(d>1e-10));

    %         fprintf('\n out_iter     rho          gapk          obj       subiter  rankX    time');
    fprintf(' %-4d %-10.2e %-10.2e %-10.2e %-10d %-10d %-10.2f\n', ...
        iter, rho, gapk, obj, subitr, rankX, ttime);

end
% get outputs
outs.rho = rho;
outs.his = outs.his(1:iter);
outs.time = outs.time(1:iter);
outs.infeas = outs.infeas(1:iter);
outs.res = outs.infeas(end);
outs.itr = iter;

xsol  = d1*Q1(2:end);

f_sol = obj_func_l1(xsol, A, b);
f_sol2 = obj_func_l1(-xsol, A, b);
if f_sol > f_sol2
    outs.fobj = f_sol2;
    xsol = -xsol;
else
    outs.fobj = f_sol;
end

ed_time = toc(tstart);

% outs.infeas = max(abs( (xsol.*xsol).^(1/2) - 1 ) ); 
outs.time = [outs.time;ed_time];
outs.his = [outs.his; outs.fobj];
outs.infeas = [outs.infeas; max(abs( (xsol.*xsol).^(1/2) - 1 ) )];

% function h = env(x, del)
% 
% z = prox(x, del);
% h = sum(sum(abs(z))) + .5*(norm(x - z,'fro')^2)./del;
% 
% end
% 
% function dh = denv(x, del)
% z = prox(x, del);
% 
% dh = (x - z)./del;
% 
% end
end

% [EOF]