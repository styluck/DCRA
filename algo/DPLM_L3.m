function [B, Wg] = DPLM_L3(Y, nbits,opt)
% optimize the supervised hashing problem by DPLM.
% B : learned binary codes.
% F : learned Hash function.
% G : learned classification matrix.

tol = opt.tol; 
nu = (1/opt.lambda);

seed = 73; 
rng(seed, 'twister');
[n,d] = size(Y);
B = sign(randn(n,nbits));

Y = nbits*Y;
 

% init B
Wg0 = zeros(nbits,d);
i = 0;

while i < opt.mitr
    i=i+1;
    
%     fprintf('Iteration  %03d: \n',i)
    % G-step
    Wg = RRC(B, Y, 1); 
    
    conv_w = norm(Wg-Wg0,'fro');
    if conv_w < tol*norm(Wg0,'fro')
        break
    end
    Wg0 = Wg;
    
    % B-step
    for ix_B = 1:opt.mitr_inner
        B0 = B;
        a = sign(B*Wg - Y)*Wg';
%         b = (opt.delta/(n^2))*B*(B'*B);
%         c = (opt.rho/(n^2))*repmat(sum(B),n,1);
        grad_p = a; % + b + c;
        
        B = sign(B - nu*grad_p);
        
        conv_B = norm(B-B0,'fro');
%         fprintf('convB: %2.2f\n', norm(B0,'fro'));
        if conv_B < tol*norm(B0,'fro')
           break
        end

    end    

%    fprintf('\n');
end

% F-step

