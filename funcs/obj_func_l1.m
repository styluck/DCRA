function [f, g] = obj_func_l1(x, A, b)

M = A*x - b;
f = sum(sum(abs(M)));

if nargout > 1
    g = A'*sign(M);

end

% [EOF]