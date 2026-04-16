function V = cal_A2(X, A)

V = .5*A*(X(2:end,1) + X(1,2:end)');


% [EOF]