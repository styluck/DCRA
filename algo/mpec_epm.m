function [x,outs] = mpec_epm(A,b, ini,opt)
% This program solves the following optimization problem:
% min ||Ax-b||_1, s.t. x \in {-1,1}^n

% parameters
maxitr = 50;%opt.mitr_inner;
maxsubitr = 10;
record = 1;
T = 10; % Update the penalty in every T iterations
rho = 0.01;

Lip = svds(A,1);
n = size(A,2);
% initialization
x = zeros(n,1); % ini; % 
v = 0; %  We initialize v0 to 0.
x_best = x;
f_min = inf;
outs.his = [];
outs.time = [];
outs.infeas = [];
t0 = tic;
if (record)
    fprintf('%-4s %-10s %-10s %-10s %-10s %-10s %-10s\n', ...
        'itr', 'subitr', 'fval', 'fbest', 'error', 'beta', 'time');
    t1 = tic;
end

for iter = 1:maxitr
    

    % solve the x-subproblem
    for subitr = 1:maxsubitr 
        %%%
        Axb = A*x - b;
        grad = A'*sign(Axb) - rho*(v);
        %%%
        xt = x;
        x = max(min(x - grad/Lip, 1), -1); %project x-grad/Lip onto [-1, 1]
        if(subitr>5 && norm(x-xt)/norm(x)<1e-5),break;end
    end

    % solve the v-subproblem
    v = sqrt(n)*(x)/norm(x);
    if iter == 1
        outs.v = v;
    end


    dist = n - dot(x,v);
    error = dist;%max(abs(1-(x).*(v)));

    f_cur = objfunc(x,A,b);
    outs.his = [outs.his;f_cur];
    outs.time = [outs.time; toc(t0)];
    outs.infeas = [outs.infeas; max(abs((x.*x).^(1/2) - 1 ))];
    if(f_min>f_cur)
        f_min = f_cur;
        x_best = x;
    end

    % recorder
    if (record) && rem(subitr, 5) == 0
        t2 = toc(t1);
        %         fprintf('\n ***********************************');
        fprintf('%-4d %-4d %-10.2e %-10.2e %-10.2e %-10.2e %-10.1f\n', ...
            iter, subitr, f_cur, f_min, error, rho, t2);

        %         fprintf('\n ***********************************');
    end

    if(~mod(iter,10))
        rho = min(rho*sqrt(T), 2*Lip);
    end
    if(iter>10&&error<0.01),break;end

end

x = sign(x_best);
outs.fobj = objfunc(x,A,b);
outs.mitr = iter;
end

function f = objfunc(x, A, b)

M = A*x - b;
f = sum(sum(abs(M)));
end





