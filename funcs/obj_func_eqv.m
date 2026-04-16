function [f, g] = obj_func_eqv(X, A, b)

M = cal_A(X, A) - b;

f = norm(M, 1);

if nargout > 1
    g = dual_A(sign(M), A);
end
% [EOF]