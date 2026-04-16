function [Atil, btil, clin, const0] = convert01_to_pm1_problem(A, y, lambda)
% Convert min ||A x - y||_1 + lambda||x||_1, x in {0,1}^N
% to    min ||Atil z - btil||_1 + clin' z,  z in {-1,1}^N
% where x = (z+1)/2. const0 = (lambda/2)*N is a drop-able constant.

    N    = size(A,2);
    Atil = 0.5 * A;
    btil = y - 0.5 * (A * ones(N,1));
    clin = (lambda/2) * ones(N,1);
    const0 = (lambda/2) * N; % add back if you need exact objective value
end
