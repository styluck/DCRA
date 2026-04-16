function [B, W] = alt_min(Y, nbits, method, opt)
% optimize the supervised hashing problem by DPLM.
% B : learned binary codes. [B, W] = alt_min(y, r, method, delta, mitr, mitr_inner) 
% F : learned Hash function. G,F,
% G : learned classification matrix.

tol = opt.tol; 
opt.nu = (1/opt.lambda);
delta = opt.delta;
opt.r = nbits;
mitr_inner = opt.mitr_inner;
Y = nbits*Y;
[d, n] = size(Y);

seed = 73; 
rng(seed, 'twister');
B = sign(randn(n,nbits));     % random {-1, 1} initialization
B=B';
% init B
Wg0 = zeros(nbits,d); 
 
for k = 1:opt.mitr
     
    % Update W using subgradient method
    W = RRC(B', Y', delta);  
    conv_w = norm(W - Wg0,'fro');
    if conv_w < tol*norm(Wg0,'fro')
        break
    end
    Wg0 = W;
    
    % Update X using external DCRA method (not implemented here)    
    switch method
        case 1
            B = sub_gradient(B, W, Y, opt);
        case 2
            [B, ~] = mpec_epm(W', Y, B, opt);
    end

end
B=B';
% f = sum(sum(abs(W'*B - Y))) + delta*norm(W,2)^2 1.1409e+06
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


%[EOF]