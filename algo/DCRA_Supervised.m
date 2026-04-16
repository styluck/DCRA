function [G,F,B] = DCRA_Supervised(X,Y,B,nbits,opts_pgm)
% X= feaTrain;
% y= traingnd;
%
% optimize the supervised hashing problem by BMPG.
% B : learned binary codes.
% F : learned Hash function.
% G : learned classification matrix.


tol = 1e-6;
[n,r] = size(B);
mu = opts_pgm.mu;
gmma = .85; % moreau envelop param
sqn = 1/sqrt(n);
M = restrictedstiefelfactory(n, r);

% label matrix N x c
% if isvector(y)
%     if nnz(y) < length(y)
%     y=y+1;
%     end
%     Y = sparse(1:length(y), y, 1); Y = full(Y);
% else
%     Y = y;
% end
Y = nbits*Y;

G = [];
% init B
Wg0 = zeros(nbits,size(Y,2));

i = 0;
while i < opts_pgm.maxItr
    i=i+1;
    % W-step
    Wg = RRC(sign(B), Y, 1);
    G.W = Wg;
    conv_w = norm(Wg-Wg0,'fro');
    fprintf('convW: %2.2f\n', conv_w);
    if conv_w < tol*norm(Wg0,'fro')
        break
    end    
    Wg0 = Wg;
    
    % B-step    
    B0 = B;
    Z = orth(B - repmat(sum(B),n,1)/n);

%     B = BMPG_BB(Z, @obj_func_BIP, @moreau_hc, @pen_hc, opts_pgm, Wg, Y);
    B = nmRGD(Z, @func, M, opts_pgm);

    conv_B = norm(B-B0,'fro'); % norm(B'*B - eye(nbits)*n) %  norm(B'*ones(n,1))
    fprintf('convB: %2.2f\n', conv_B);
    if conv_B < tol*norm(B0,'fro') 
        break
    end
   fprintf('\n');
end

% F-step
fprintf('obj val: %2.2f\n', obj_func_BIP(B,Wg,Y))
B = sign(B);
WF = RRC(X, B, 1e-2);
F.W = WF;
% B = sign(B);

    function [f, g] = func(X)
        [f1, g1] = obj_func_BIP(X, Wg, Y);
%         fprintf('normWg:%2.2f\n',Wg);
        [f2, g2] = moreau_hc(X,sqn,mu,gmma);
        f = f1 + f2;
        g = g1 + g2;
    end
end

% [EOF]