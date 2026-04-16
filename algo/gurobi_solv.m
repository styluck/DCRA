function [x_opt, result] = gurobi_solv(A, b,ini, params)

% Gurobi parameters (optional)
% params.outputflag = 1; % Display output
params.outputflag = 0; % 关闭所有输出信息
params.LogToConsole = 0; % 关闭控制台日志
% params.Callback = @mycallback; % Display output
% params.TimeLimit = 30;
addpath E:\gurobi1103\win64\matlab
[n,m] = size(A);
f = [zeros(m,1);ones(n,1)];

C = 2*A;
d = A*ones(m,1) + b;

% Define the Gurobi model
% model = gurobi(model);
model.modelsense = 'min';
model.start = [ini;zeros(n,1)];
% x0 = [(x_opt+1)/2;y;abs(y)];


% Decision variables: x (binary), y (continuous), and z (continuous);
model.vtype = [repmat('B', m, 1); repmat('C',n,1)];

% Objective: minimize z'*1
model.obj = f;

% Constraints:
% model.A = sparse([C, -eye(n), zeros(n)]); % A must be sparse
% model.rhs = full([d(:)]); % rhs must be dense
% model.sense = [repmat('=',n,1)];

model.A = sparse([C, -eye(n); -C, -eye(n)]); % A must be sparse
model.rhs = full([d(:);d(:)]); % rhs must be dense
model.sense = [repmat('<',2*n,1)];
% 
% y_index = (m+1):(n+m);
% z_index = (n+m+1):(2*n+m);
% for i = 1:n
% % i=1;
% model.genconabs(i).resvar = z_index(i);
% model.genconabs(i).argvar = y_index(i);
% end

% Solve the problem
result = gurobi(model, params);

% Extract the solution
y_opt = result.x(1:m);
x_opt = 2 * y_opt - 1; % Convert back to {-1, 1}
% y_opt = outs.x(1:m);
% x_opt = 2 * y_opt - 1; % Convert back to {-1, 1}
% Display results
% disp('Optimal solution x:');
% disp(x_opt);


M = A*x_opt - b;
result.fobj = sum(sum(abs(M)));
disp('Optimal objective value:');
disp(result.fobj); 
end

function status = myCallback(model, where)
    global iterationData;
    status = 0;  % Default status
    
    % Check if the callback is at the simplex or MIP stage
    if where == model.CB_MIP
        % Retrieve the objective value and runtime
        objVal = model.cbGet(model.CB_MIP_OBJBST); % Current best objective
        bestBound = model.cbGet(model.CB_MIP_OBJBND); % Current best bound
        runtime = model.cbGet(model.CB_RUNTIME); % Elapsed time
        
        % Append data to the iteration data
        iterationData = [iterationData; objVal, bestBound, runtime];
    end
end

