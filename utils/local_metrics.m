function [Obj,MSE,Res,L0] = local_metrics(z, A, y, lambda, x0)
    x = (sign(z)+1)/2;
    Res = sum(abs(A*x - y));
    L0  = nnz(x);
    Obj = Res + lambda * sum(x);
    MSE = norm(x - x0)^2 / numel(x0);
%     metrics = [Obj,MSE,Res,L0];
end