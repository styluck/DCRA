% demostration code to generate a table result with different methods and
% different dimensions. 
clear; close all;

% Define range of dimensions for testing
r_values = [200 500];%, 300, 500, 1000, 2000, 3000]; % Number of rows in A , 200, 300, 500, 1000, 2000, 3000
n_values = [50, 100];%, 200, 300, 500, 1000, 2000];  % Number of columns in A , 200, 300, 500, 1000, 2000

% Number of random tests per configuration
notest = 1;

% Seed for reproducibility
randstate = 100;
randn('state', double(randstate));
rand('state', double(randstate));
 
% Initialize result table with headers
result_table = {'m', 'n', ...
    'Avg_obj_DCRA', 'Avg_obj_EPM', 'Avg_obj_Gurobi', 'Avg_obj_ADM',...
    'Avg_Diff_EPM', 'Avg_Diff_Gurobi', 'Avg_Diff_ADM', ...
    'Win_Rate_EPM', 'Win_Rate_Gurobi', 'Win_Rate_ADM', ...
    'Win_Value_EPM', 'Win_Value_Gurobi', 'Win_Value_ADM', ...
    'Time_DCRA', 'Time_EPM', 'Time_Gurobi', 'Time_ADM', ...
    }; % Added time columns
row_idx = 2; % Start from the second row for data
% A = zeros(4, 10); 
% Loop through different dimensions
for r = r_values
    for n = n_values
        % Initialize result storage for the current dimension
        results = zeros(notest, 7); % Differences for 3 algorithms compared to mpec_epm
        times_dcra = zeros(notest, 1); % Computational time for DCRA
        times_gurobi = zeros(notest, 1); % Computational time for gurobi_solv
        times_adm = zeros(notest, 1); % Computational time for mpec_adm
        times_epm = zeros(notest, 1); % Computational time for mpec_epm

        for n_test = 1:notest
            % Generate random data for the problem

            A = randn(n, r);
            b = randn(n, 1);
% N = 500; alpha = 0.3; rho = .2; mu = 200; lambda = .05; seed = 1;
% [A, x0, y, ~] = gen_binary_cs_instance(N, alpha, rho, mu, 'seed', seed);
% 
% % ---------- Convert problem to {-1,1} domain (A' = 0.5 A, b' = y - 0.5 A 1, c_z on z)
% [A, b, clin, const0] = convert01_to_pm1_problem(A, y, lambda);  %#ok<ASGLU>
% 
% M   = size(A,1);                  % number of rows (residual block length)
% Nz  = size(A,2);                  % number of cols / dim of z
% c_z = (lambda/2) * ones(Nz,1);    % linear term coefficients on z
% 
% % (Optional) augmented matrices if your DCRA expects f(Ā z - b̄)
% A = [A; speye(Nz)];         % (M+N) x N
% b = [b; zeros(Nz,1)];       % (M+N) x 1
% [n, r] = size(A);
            % Initialization for DCRA
            m = 2; % max(2, round(0.2 * n));% 2; %max(2, round(0.2 * m)); % 30;% Select m_v in [2, m/5]
            V0 = randn(m, r + 1);
%             [P, D, Q] = svd(V0,'econ');
%             P1 = P(:,1);
%             Q1 = Q(:,1);    
%             d = diag(D); d1 = d(1);
            X0 = zeros(r,1);%double(d1*Q1(2:end)>0); 


            % Measure time for mpec_epm algorithm (baseline)
            tic;
            [x_opt_epm, outs_epm] = mpec_epm(A, b,X0);
            times_epm(n_test) = toc;
            f_epm = outs_epm.fobj;

            % Measure time for DCRA algorithm
            tic;
            [xsol_dcra, outs_dcra] = DCRA(A, b, V0, @env_l1);
            times_dcra(n_test) = toc;
            f_dcra = outs_dcra.fobj; 
            % Measure time for gurobi_solv algorithm
            tic;
            [x_opt_gurobi, outs_gurobi] = gurobi_solv(A, b,X0);
            times_gurobi(n_test) = toc;
            f_gurobi = outs_gurobi.fobj;

            % Measure time for mpec_adm algorithm
            tic;
            [x_opt_adm, outs_adm] = mpec_adm(A, b,X0);
            times_adm(n_test) = toc;
            f_adm = outs_adm.fobj;

            % Calculate differences relative to DCRA
            v1 = f_dcra - f_epm;     % mpec_epm vs. DCRA 
            v2 = f_dcra - f_gurobi;  % gurobi_solv vs. DCRA
            v3 = f_dcra - f_adm;     % mpec_adm vs. DCRA
            v4 = f_dcra;
            v5 = f_epm;
            v6 = f_gurobi;
            v7 = f_adm;

            results(n_test, :) = [v1, v2, v3, v4, v5, v6, v7];    

        end

        % Calculate metrics for all algorithms
        avg_diff_epm = mean(results(:, 1));
        avg_diff_gurobi = mean(results(:, 2));
        avg_diff_adm = mean(results(:, 3));

        avg_obj_dcra = mean(results(:, 4));
        avg_obj_epm = mean(results(:, 5));
        avg_obj_gurobi = mean(results(:, 6));
        avg_obj_adm = mean(results(:, 7));

        winrate_epm = sum(results(:, 1) < 0) / notest;
        winrate_gurobi = sum(results(:, 2) < 0) / notest;
        winrate_adm = sum(results(:, 3) < 0) / notest;

        winvalue_epm = mean(results(results(:, 1) < 0, 1));
        winvalue_gurobi = mean(results(results(:, 2) < 0, 2));
        winvalue_adm = mean(results(results(:, 3) < 0, 3));

        % Calculate average time for each algorithm
        avg_time_dcra = mean(times_dcra);
        avg_time_gurobi = mean(times_gurobi);
        avg_time_adm = mean(times_adm);
        avg_time_epm = mean(times_epm);
        
        % Save results in the table
        result_table{row_idx, 1} = r;                    % Rows of A
        result_table{row_idx, 2} = n;                    % Columns of A

        result_table{row_idx, 3} = avg_obj_dcra;        % obj for DCRA
        result_table{row_idx, 4} = avg_obj_epm;         % obj for mpec_epm
        result_table{row_idx, 5} = avg_obj_gurobi;      % obj for gurobi
        result_table{row_idx, 6} = avg_obj_adm;         % obj for mpec_adm

        result_table{row_idx, 7} = avg_diff_epm;        % Avg. Diff against mpec_epm
        result_table{row_idx, 8} = avg_diff_gurobi;      % Avg. Diff against gurobi_solv
        result_table{row_idx, 9} = avg_diff_adm;         % Avg. Diff against mpec_adm

        result_table{row_idx, 10} = winrate_epm;         % Win Rate against mpec_epm
        result_table{row_idx, 11} = winrate_gurobi;       % Win Rate against gurobi_solv
        result_table{row_idx, 12} = winrate_adm;         % Win Rate against mpec_adm

        result_table{row_idx, 13} = winvalue_epm;        % Win Value against mpec_epm
        result_table{row_idx, 14} = winvalue_gurobi;      % Win Value against gurobi_solv
        result_table{row_idx, 15} = winvalue_adm;        % Win Value against mpec_adm

        result_table{row_idx, 16} = avg_time_dcra;       % Time for DCRA
        result_table{row_idx, 17} = avg_time_epm;        % Time for mpec_epm
        result_table{row_idx, 18} = avg_time_gurobi;     % Time for gurobi_solv
        result_table{row_idx, 19} = avg_time_adm;        % Time for mpec_adm

        row_idx = row_idx + 1;
    end
end

% Convert result table to a cell array for better display
disp('Test Results:');
disp(result_table);
save('result2.mat','result_table')
writecell(result_table, 'result_table2.xlsx');
