function V = dual_A(x, A)

Ax = .5*A'*x;
n = length(Ax);
V = zeros(n+1);
V(1,2:end) = Ax';
V(2:end,1) = Ax;

% [EOF]