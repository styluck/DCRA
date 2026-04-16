function [x,outs] = mpec_epm_bcs(A,b,c_z,ini,opt)
% Solve:  min ||A x - b||_1 + c_z' * x,   s.t. x in {-1,1}^n
% Inputs:
%   A (M x n), b (M x 1), c_z (n x 1), ini (n x 1), opt (struct, optional)

    % ---------------- Parameters
    if nargin < 5, opt = struct(); end
    maxitr    = get_opt(opt,'maxitr',50);
    maxsubitr = get_opt(opt,'maxsubitr',100);
    record    = get_opt(opt,'record',1);
    T         = get_opt(opt,'T',10);            % penalty update interval
    rho       = get_opt(opt,'rho',0.01);        % penalty weight (starts small)
    Lip       = get_opt(opt,'Lip',norm(A));     % Lipschitz estimate (||A||_2)
    n         = size(A,2);

    % ---------------- Initialization
    x      = ini(:);
    v      = zeros(n,1);                        % MPEC auxiliary
    x_best = x;
    f_min  = inf;

    outs.his   = [];
    outs.time  = [];
    outs.infeas= [];
    t0 = tic;

    if record
        fprintf('%-4s %-10s %-10s %-10s %-10s %-10s %-10s\n', ...
            'itr','subitr','fval','fbest','error','beta','time');
        t1 = tic;
    end

    % ---------------- Main loop
    for iter = 1:maxitr

        % ----- x-subproblem (projected subgradient / PG on box [-1,1])
        for subitr = 1:maxsubitr
            Axb  = A*x - b;
            grad_l1 = A' * sign(Axb);           % subgrad of ||A x - b||_1
            grad_lin= c_z;                      % grad of linear term
            grad_pen= -rho * v;                 % penalty driving towards |x|=1
            grad   = grad_l1 + grad_lin + grad_pen;

            x_old = x;
            x     = x - grad / Lip;
            x     = max(min(x, 1), -1);         % project to box [-1,1]

            if subitr>5 && norm(x - x_old)/max(1,norm(x)) < 1e-5
                break;
            end
        end

        % ----- v-subproblem (normalize x onto sphere to enforce |x_i|≈1)
        if norm(x) == 0
            v = sqrt(n) * ones(n,1) / sqrt(n);
        else
            v = sqrt(n) * x / norm(x);
        end
        if iter == 1, outs.v = v; end

        % ----- diagnostics
        dist  = n - dot(x,v);                   % MPEC distance term
        error = dist;

        f_cur = objfunc_bcs(x,A,b,c_z);
        outs.his    = [outs.his;   f_cur];
        outs.time   = [outs.time;  toc(t0)];
        outs.infeas = [outs.infeas; max(abs(abs(x) - 1))];

        if f_cur < f_min
            f_min  = f_cur;
            x_best = x;
        end

        % ----- recorder
        if record && rem(subitr,5)==0
            t2 = toc(t1);
            fprintf('%-4d %-4d %-10.2e %-10.2e %-10.2e %-10.2e %-10.1f\n', ...
                iter, subitr, f_cur, f_min, error, rho, t2);
        end

        % ----- penalty schedule & stopping
        if ~mod(iter,10)
            rho = min(rho*sqrt(T), 2*Lip);
        end
        if iter>10 && error<1e-2
            break;
        end
    end

    % ----- finalize on the discrete set
    x = sign(x_best);
    outs.fobj = objfunc_bcs(x,A,b,c_z);
    outs.mitr = iter;
end

function f = objfunc_bcs(x,A,b,c_z)
    r = A*x - b;
    f = sum(abs(r)) + sum(c_z.* x);
end

function v = get_opt(opt, field, default)
    if isfield(opt,field), v = opt.(field); else, v = default; end
end
