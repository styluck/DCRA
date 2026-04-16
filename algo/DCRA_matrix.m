function [xsol, outs] = DCRA_matrix(A, b, V, OPTIONS)
%% ***************************************************************
% filename: QAfactor
% to solve the factorized form of the penalty problem of (3) induced by
% the DC constraint
% \underset{x\in \{-1, 1\}^n}{\min} \|Ax-b\|_1 
%% ***************************************************************

if nargin < 4 || isempty(OPTIONS)
    OPTIONS = struct();
end

if nargin < 3 || isempty(V)
    m = size(A, 2);
    p = m + 1;
    N = size(b, 2);

    V = randn(50, p, N);
else
    [k, p, N] = size(V);
end

prox = @prox_l1;
V = retr_ob(V); % generate the initial point V\in Ob(n,r)

sn = sqrt(p);
if isfield(OPTIONS,'gaptol');    gap_tol  = OPTIONS.gaptol;    else; gap_tol =1e-8; end
if isfield(OPTIONS,'objtol');    obj_tol  = OPTIONS.objtol;    else; obj_tol =1e-8; end
if isfield(OPTIONS,'xtol');      xtol     = OPTIONS.xtol;      else; xtol =1e-4*sn; end
if isfield(OPTIONS,'ftol');      ftol     = OPTIONS.ftol;      else; ftol =1e-4*sn; end
if isfield(OPTIONS,'maxiter');   maxiter  = OPTIONS.maxiter;   else; maxiter =1e3;  end
if isfield(OPTIONS,'submxitr');  submxitr = OPTIONS.submxitr;  else; submxitr =30;  end
if isfield(OPTIONS,'printyes');  printyes = OPTIONS.printyes;  else; printyes =1;   end
if isfield(OPTIONS,'recordALM'); recordALM = OPTIONS.recordALM;else; recordALM =0;  end

% ***************** The factor_penalty method *****************************
if (printyes)
    fprintf('\n **********************************************************************************');
    fprintf('\n %-4s %-12s %-10s %-4s %-12s %-12s %-12s\n', ...
        'iter', 'rho', 'gapk', 'obj', 'subiter', 'rankX', 'time');

end

% ************************* Main Loop *********************************
tstart = tic;

rho = 5; % initial penalty param
sigma = 1.2; % increment of the penalty param
del = 1e-4;

P1 = zeros(k, N);
Q1 = zeros(k, N);
[P, D, Q] = svd(V,'econ');
P1 = P(:,1);
Q1 = Q(:,1);

d = diag(D);

clear P Q D

Gmma = -2*(d(1)*P1)*Q1';
Lf = 1 + norm(A);

outs.time = zeros(maxiter, 1);
outs.his = zeros(maxiter, 1);
outs.infeas = zeros(maxiter, 1);
objold = inf;
for iter = 1:maxiter

    % solve the augmented lagrangian subproblem
    if (recordALM)
        fprintf('%-4s %-10s %-10s %-10s %-10s %-10s %-10s\n', ...
            'itr', 'subitr', 'g(x)', 'fval', 'dif_Lmb', 'beta', 'time');

        tstart = tic;
    end

    obj_list = zeros(submxitr, 1);
    Vold = V;
    vv = V(:,1)' * V(:,2:end);
    AVb = cal_A(vv, A) - b;


    for subitr = 1:submxitr
        % solve the V-subproblem by the Riemannian gradient descent method

        SCALAR = 1/(2*rho + Lf);

        OUTER = Lf*Vold - rho*Gmma - 2*Vold*dual_A(denv(AVb, del), A);
        V = retr_ob(V, SCALAR*OUTER);
        VV = V'*V;
        AVb = cal_A(VV, A) - b;

        [P, D, Q] = svd(V,'econ');
        P1 = P(:,1);
        Q1 = Q(:,1);
        d = diag(D);
        Gmma = -2*(d(1)*P1)*Q1';

        gsubiter = 0;

        % termination condition for ALM:
        if (subitr >= 10) && (  ...
                (abs(obj - obj_list(subitr - 9))/(1 + abs(obj)) <= ftol) || ...
                (norm(Vold - V,'fro') < xtol)  )
            break;
        end

        d1 = d(1);  % d is calculated when augLagfunc is called in OptManiMulitBallGBB
        gapk = sum(d.^2)-d1^2; % ||V||_F^2 - ||V||_2^2

        % recorder
        if (recordALM) && rem(subitr,5) == 0
            ttime = toc(tstart);
            
            %         fprintf('\n **********************************************************************************');
            fprintf('%-4d %-4d %-10.2e %-10.2e %-10.2e %-10.2e %-10.1f\n', ...
                subitr, gsubiter, gapk, obj, normAVbW, bta, ttime);

            %         fprintf('\n **********************************************************************************');
        end

        %         obj = obj_func_eqv(VV, A, b);
        obj = env(AVb, del);
        obj_list(subitr) = obj + rho*gapk;

        Vold = V;
    end

    ttime = toc(tstart);
    outs.time(iter) = ttime;
    outs.his(iter) = obj;
    outs.infeas(iter) = gapk;
    if (printyes)&&(mod(iter,5)==0)

        
        rankX= length(d(d>1e-10));
        %         fprintf('\n out_iter     rho          gapk          obj       subiter  rankX    time');
        fprintf(' %-4d %-10.2e %-10.2e %-10.2e %-10d %-10d %-10.2f\n', ...
            iter, rho, gapk, obj, subitr, rankX, ttime);

    end

    % stopping criterion for outer iterate
    if gapk<gap_tol && abs(objold-obj_list(subitr))/max(1,abs(obj_list(subitr)))<obj_tol
        break;

    else
        rho = min(sigma*rho,1.0e+6); % increase the penalty param
    end
    % ****************

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

function h = env(x, del)

z = prox(x, del);
h = sum(sum(abs(z))) + .5*(norm(x - z,'fro')^2)./del;

end

function dh = denv(x, del)
z = prox(x, del);

dh = (x - z)./del;

end
end

% [EOF]