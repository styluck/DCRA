
function y = local_smooth(x, w)
% Simple moving median (or mean if medfilt1 not available).
    x = x(:)';  w = max(1, round(w));
    if exist('medfilt1','file') == 2
        y = medfilt1(x, w, 'omitnan','truncate');
    else
        % moving average fallback
        k = ones(1,w)/w;
        y = conv(x, k, 'same');
    end
end