function [f, g] = obj_func_inf(x, A, b)

M = A*x - b;
f = norm(M, inf);

if nargout > 1
    g = A'*sign(M);

end

% [EOF]