function [x_prox, Act_set, Inact_set] = prox_l1(b, lambda)

a = abs(b) - lambda;

Act_set = (a > 0);

x_prox = (Act_set.*sign(b)).*a;

if nargout==3
    Inact_set= (a <= 0);
end

end

