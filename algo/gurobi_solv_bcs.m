function [x_opt, result] = gurobi_solv_bcs(A, b, ini, lambda, params)
% Solve:  min ||A x - b||_1 + lambda * ||x||_1,   s.t. x in {0,1}^n
% Variables: x (binary n-by-1), z (continuous m-by-1) s.t. z >= |A x - b|

    if nargin < 5 || isempty(params)
        params = struct();
        params.outputflag = 0;           % quiet by default
        params.LogToConsole = 0;
    else
        % keep your defaults unless caller overrides
        if ~isfield(params,'outputflag'),   params.outputflag   = 0; end
        if ~isfield(params,'LogToConsole'), params.LogToConsole = 0; end
    end

    if nargin < 4, lambda = 0; end

    addpath E:\gurobi1103\win64\matlab

    % --- sizes
    [m, n] = size(A);                     % m rows (residuals), n vars (binary)

    % --- objective coefficients for [x; z]
    %     obj = lambda*sum(x) + sum(z)
    model.obj        = [lambda*ones(n,1); ones(m,1)];
    model.modelsense = 'min';

    % --- constraints: z >= A x - b  and  z >= -(A x - b)
    %     -> [-A  I]*[x;z] >= -b
    %        [ A  I]*[x;z] >=  b
    A1 = [-A,  speye(m)];
    A2 = [ A,  speye(m)];
    model.A     = sparse([A1; A2]);
    model.rhs   = [ -b;  b ];
    model.sense = [ repmat('>', m,1); repmat('>', m,1) ];

    % --- variable types and bounds
    model.vtype = [repmat('B', n,1); repmat('C', m,1)];
    model.lb    = [zeros(n,1); zeros(m,1)];     % x in [0,1], z >= 0
    model.ub    = [ones(n,1);  inf(m,1)];

    % --- warm start (optional): ini may be {-1,1} or {0,1}; sanitize
    if nargin >= 3 && ~isempty(ini)
        x0 = ini(:);
        if any(x0 < 0)                      % likely {-1,1} init
            x0 = (sign(x0)+1)/2;
        end
        x0 = min(max(round(x0),0),1);
        model.start = [x0; zeros(m,1)];
    end

    % --- solve
    result = gurobi(model, params);

    % --- parse solution
    x_opt = [];
    if isfield(result,'x')
        sol   = result.x;
        x_opt = sol(1:n);                   % binary solution in {0,1}
    else
        warning('Gurobi returned status: %s', result.status);
    end

    % --- report objective in original form (for convenience)
    if ~isempty(x_opt)
        resid = A*x_opt - b;
        result.fobj = sum(abs(resid)) + lambda * sum(x_opt);
    else
        result.fobj = NaN;
    end
end
