function [x,outs] = mpec_adm_bcs(A,b,c_z,ini,opt)
% Solve:  min ||A x - b||_1 + c_z' * x,   s.t. x in {-1,1}^n
% Inputs:
%   A (m x n), b (m x 1), c_z (n x 1), ini (n x 1), opt (struct, optional)

% ---------------- parameters (with your defaults)
if nargin < 5, opt = struct(); end
maxitr     = get_opt(opt,'maxitr',50);        % outer iters
maxsubitr  = get_opt(opt,'maxsubitr',100);     % inner PG steps
record     = get_opt(opt,'record',1);
T          = get_opt(opt,'T',10);             % penalty update interval
rho        = get_opt(opt,'rho',0);            % ALM multiplier
alpha      = get_opt(opt,'alpha',1e-4);       % ALM penalty growth
threshold  = get_opt(opt,'threshold',50);
tol        = get_opt(opt,'tol',0.01);

n      = size(A,2);
max_mm = norm(A);                             % spectral-norm estimate

% ---------------- initialization
x      = ini(:);
v      = zeros(n,1);
x_best = x;
f_min  = inf;

outs.his    = [];
outs.time   = [];
outs.infeas = [];
t0 = tic;

if record
    fprintf('%-4s %-10s %-10s %-10s %-10s %-10s %-10s %-10s\n', ...
        'itr','subitr','fval','fbest','error','beta','alpha','time');
    t1 = tic;
end

% ---------------- main loop
for iter = 1:maxitr

    % local Lipschitz estimate (same as your code)
    Lip = max_mm + alpha * 4 * norm(x)^2;

    % ----- x-subproblem: projected (sub)gradient on box [-1,1]
    for subitr = 1:maxsubitr
        r        = A*x - b;
        grad_l1  = A' * sign(r);              % subgrad of ||A x - b||_1
        grad_lin = c_z;                        % grad of c_z' * x
        grad_pen = -rho * v - 2*alpha*(n - sum(x.*v));  % as in your code

        grad = grad_l1 + grad_lin + grad_pen;

        xt = x;
        x  = x - grad / Lip;
        x  = max(min(x, 1), -1);              % project to [-1,1]

        if subitr > 5 && norm(x-xt)/max(1,norm(x)) < 1e-5
            break;
        end
    end

    % ----- v-subproblem (your rank-1 QP projection)
    v = QP_rank1(x, -n*(x) - rho/alpha*(x), sqrt(n), 0);

    % ----- ALM updates / diagnostics
    dist  = n - sum(x.*v);
    rho   = rho + alpha * dist;
    error = max(0, dist);

    f_cur = objfunc_bcs(x,A,b,c_z);
    outs.his    = [outs.his; f_cur];
    outs.time   = [outs.time, toc(t0)];
    outs.infeas = [outs.infeas; max(abs(abs(x) - 1))];

    if f_cur < f_min
        f_min  = f_cur;
        x_best = x;
    end

    % ----- recorder
    if record && rem(subitr,5)==0
        t2 = toc(t1);
        fprintf('%-4d %-4d %-10.2e %-10.2e %-10.2e %-10.2e %-10.2e %-10.1f\n', ...
            iter, subitr, f_cur, f_min, error, rho, alpha, t2);
    end

    % ----- penalty schedule
    if rho <= threshold
        if ~mod(iter, T), alpha = alpha * sqrt(10); end
    else
        rho = threshold;
    end

    % ----- stopping
    if iter > 30 && error < tol, break; end
end

% ----- finalize on {-1,1}
x = sign(x_best);
outs.fobj = objfunc_bcs(x,A,b,c_z);
outs.mitr = iter;
end

% ---------- helpers
function f = objfunc_bcs(x,A,b,c_z)
    r = A*x - b;
    f = sum(abs(r)) + sum(c_z.* x);
end

function v = get_opt(opt,field,default)
    if isfield(opt,field), v = opt.(field); else, v = default; end
end
