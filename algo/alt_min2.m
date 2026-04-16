function [B, W] = alt_min2(Y, nbits, method, opt)
% optimize the supervised hashing problem by DPLM.
% B : learned binary codes. [B, W] = alt_min(y, r, method, delta, mitr, mitr_inner) 
% F : learned Hash function. G,F,
% G : learned classification matrix.

tol = opt.tol; 
opt.nu = (1/opt.lambda);
delta = opt.delta;
opt.r = nbits;
% mitr_inner = opt.mitr_inner;
Y = nbits*Y;
[d, n] = size(Y);
opt.n = n;
seed = 13; 
rng(seed, 'twister');
B = sign(randn(n,nbits));     % random {-1, 1} initialization
B=B';
% init B
Wg0 = zeros(nbits,d); 

L = n * sqrt(d * nbits) + delta;
eta = 1 / L;

for k = 1:opt.mitr
     
    % Update W using subgradient method
    W = RRC(B', Y', delta); 
%     W = update_W_subgrad(Y, B, delta, Wg0', eta, mitr_inner);
    conv_w = norm(W - Wg0,'fro');
    if conv_w < tol*norm(Wg0,'fro')
        break
    end
    Wg0 = W;
    
    % Update X using external DCRA method (not implemented here)    
%     switch method
%         case 1
%             B = sub_gradient(B, W, Y, opt);
%         case 2
 if method <= 3
            B = dcra_update(B, W', Y, method, opt);
 elseif method == 4
     B = sub_gradient(B, W, Y, opt);
 elseif method == 5
     B = dcra_update(B, W', Y, 5, opt);
 end

%             [B, ~] = mpec_epm(W', Y, B, opt);
%     end

%    fprintf('\n');
end
B=B';
% f = sum(sum(abs(W'*B - Y))) + delta*norm(W,2)^2 1.1409e+06
end

function X = dcra_update(X, W, Y, method, opt)
% Inputs:
%   W - matrix (d x r), plays role of A in each subproblem
%   B - matrix (d x n)
%
% Output:
%   X - matrix (n x r), binary {-1, 1}

r = opt.r;

p = r + 1;     % ambient dimension of each x_j
m_v = 5;%min(100,p);  % low-rank dimension for initialization
OPTION.maxiter =  opt.mitr_inner;
OPTION.del = 1e-4;
OPTION.rho = .005;

randstate = 100;
randn('state',double(randstate));
rand('state',double(randstate));
% method = 1; 
V0 = randn(m_v, p);    % random low-rank initialization
parfor j = 1:opt.n % parfor
 
    switch method
        case 1
            
            x_j = DCRA(W, Y(:, j), V0, @env_l1, OPTION);  % solve vector problem

        case 2
            X0 = X(:,j);
            [x_j, ~] = mpec_epm(W, Y(:, j), X0, opt);

        case 3
            X0 = X(:,j);
            [x_j, ~] = mpec_adm(W, Y(:, j), X0, opt);

        case 5
            X0 = X(:,j); 
            [x_j, ~] = gurobi_solv(W, Y(:, j), X0);

    end
    X(:, j) = x_j;         % store in j-th column
end

end


function B = sub_gradient(B, W, Y, opt)
    for ix_B = 1:opt.mitr_inner
        B0 = B;
%         a = W*sign(W'*B - Y);
%         b = (opt.delta/(n^2))*B*(B'*B);
%         c = (opt.rho/(n^2))*repmat(sum(B),n,1);
        grad_p = W*sign(W'*B - Y); % + b + c;
        
        B = sign(B - opt.nu*grad_p);
        
        conv_B = norm(B-B0,'fro');
%         fprintf('convB: %2.2f\n', norm(B0,'fro'));
        if conv_B < opt.tol*norm(B0,'fro')
           break
        end

    end
end


function W = update_W_subgrad(Y, B, delta, W0, eta, maxIter)
% Inputs:
%   B        - data matrix (d x n)
%   X        - binary matrix (n x r)
%   delta    - regularization parameter
%   W0       - initial value of W (d x r)
%   maxIter  - number of subgradient iterations
%
% Output:
%   W        - updated W after subgradient iterations
W = W0;

for t = 1:maxIter
    R = Y - W * B;                      % Residual
    G = -sign(R) * B' + delta * W;        % Subgradient

    W = W - eta * G;                     % Update
end
W = W';
end
%[EOF]