% ---------- Helper: postprocess a {-1,1} solution
function [x01, f0, mse] = post_eval(z, A, y, lambda, x0)
    if ~all(abs(z)==1)
        z = sign(z); z(z==0) = 1;
    end
    x01 = pm1_to_bin01(z);
    f0  = norm(A*x01 - y, 1) + lambda * norm(x01, 1);
    mse = norm(x01 - x0)^2 / numel(x0);
end