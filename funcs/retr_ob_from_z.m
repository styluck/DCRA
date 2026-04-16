function V = retr_ob_from_z(z, m)
    n = numel(z); %m = 2;                       % keep your rank
    V = randn(m, n+1);                         % (r x (n+1))
    V(:,1) = [1; zeros(m-1,1)];                % anchor
    V(:,2:end) = V(:,2:end) - mean(V(:,2:end),2);  % center columns
    V = retr_ob(V);                            % your existing projection
    % push first column direction toward z
    u = [1; z(:)];  u = u / norm(u);
    [P,D,Q] = svd(V,'econ'); %#ok<ASGLU>
    V = P*D*Q'; V(1,:) = u; V = retr_ob(V);
end