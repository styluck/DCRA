function x_pm = bin01_to_pm1(x01)
% BIN01_TO_PM1 Convert binary {0,1} vector to {-1,1} vector.
%
% Usage:
%   x_pm = bin01_to_pm1(x01);
%
% Input:
%   x01 : vector/matrix with entries in {0,1}
%
% Output:
%   x_pm: same size as x01, entries in {-1,1}

    % check validity (optional)
    if any(x01(:) ~= 0 & x01(:) ~= 1)
        warning('Input has values outside {0,1}.');
    end
    
    % convert
    x_pm = 2*x01 - 1;
end
