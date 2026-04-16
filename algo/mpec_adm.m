function [x,outs] = mpec_adm(A,b, ini, opt)
% This program solves the following optimization problem:
% min ||Ax-b||_1, s.t. x \in {-1,1}^n

% parameters
maxitr = 50;% opt.mitr_inner;
maxsubitr = 10;
record = 1;
T = 10; % Update the penalty in every T iterations
rho = 0;
alpha = 0.0001;
threshold = 50;
tol = 0.01;

n = size(A,2);
max_mm = svds(A,1);

% initialization
x = ini; %zeros(n,1);
v = 0;
x_best = x;
f_min = inf;
outs.his = [];
t0 = tic;
outs.time = [];
outs.infeas = [];

if (record)
    fprintf('%-4s %-10s %-10s %-10s %-10s %-10s %-10s %-10s\n', ...
        'itr', 'subitr', 'fval', 'fbest', 'error', 'beta', 'alpha', 'time');
    t1 = tic;
end

for iter = 1:maxitr
    
    Lip = max_mm + alpha *4 * norm(x)^2;

    for subitr=1:maxsubitr

        %%%
        Axb = A*x - b;
        grad = A'*sign(Axb) - rho*(v) - 2*alpha*(n - sum(x.*v));
        %%% 
        xt = x;
        x = max(min(x - grad/Lip, 1), -1); 
        if(subitr>5 && norm(x-xt)/norm(x)<1e-5),break;end
    end
    
    v = QP_rank1(x,-n*(x)-rho/alpha*(x),sqrt(n),0);
    
    dist = n - sum(x.*v);
    rho = rho + alpha * dist;
    error = max(0,dist);
    
    f_cur = objfunc(x,A,b);
    outs.his = [outs.his;f_cur];
    outs.time = [outs.time, toc(t0)];
    outs.infeas = [outs.infeas; max(abs((x.*x).^(1/2) - 1 ))];
    if(f_min>f_cur)
        f_min = f_cur;
        x_best = x;
    end
    
    % recorder
    if (record) && rem(subitr, 5) == 0
        t2 = toc(t1);
        %         fprintf('\n ***********************************');
        fprintf('%-4d %-4d %-10.2e %-10.2e %-10.2e %-10.2e  %-10.2e %-10.1f\n', ...
            iter, subitr, f_cur, f_min, error, rho, alpha, t2);

        %         fprintf('\n ***********************************');
    end
    
    if(rho<=threshold)
        if(~mod(iter,T))
            alpha = alpha * sqrt(10);
        end
    else
        rho = threshold;
    end
        
    if(iter>30 && error<tol),break;end
    
end

x = sign(x_best);
outs.fobj = objfunc(x,A,b);
outs.mitr = iter;
end

function f = objfunc(x, A, b)

M = A*x - b;
f = sum(sum(abs(M)));
end




