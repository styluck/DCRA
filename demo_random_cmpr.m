% demostration code to generate a table result with different methods and
% different dimensions. Now only compare DCRA and new_algo.
clear; close all;

% ---------------- 基本设置 ----------------
% Define range of dimensions for testing
r_values = [200 500 1000 2000];              % Number of rows in A
n_values = [50 100 500 1000];               % Number of columns in A

% Number of random tests per configuration
notest = 100;

% Seed for reproducibility
randstate = 100;
randn('state', double(randstate));
rand('state', double(randstate));

% ---------------- 结果表头 ----------------
% 这里假设你关心：目标值、残差、迭代次数、时间
% 可以根据自己需要增减列
header = { ...
    'r', 'n', ...
    'obj_old',   'obj_new', ...
    'time_old',  'time_new'};
    % 'res_old',   'res_new', ...
    % 'iter_old',  'iter_new', ...

num_configs = numel(r_values) * numel(n_values);
result_table = cell(num_configs + 1, numel(header));
result_table(1, :) = header;

row_idx = 2;
m = 2;
% ---------------- 主循环：遍历 r, n ----------------
for ir = 1:numel(r_values)
    r = r_values(ir);
    for in = 1:numel(n_values)
        n = n_values(in);

        % 用来做多次随机测试的累积
        obj_dcra_all   = zeros(notest, 1);
        obj_new_all    = zeros(notest, 1);
        res_dcra_all   = zeros(notest, 1);
        res_new_all    = zeros(notest, 1);
        iter_dcra_all  = zeros(notest, 1);
        iter_new_all   = zeros(notest, 1);
        time_dcra_all  = zeros(notest, 1);
        time_new_all   = zeros(notest, 1);

        for itest = 1:notest
            % -------- 生成测试问题（示例）--------
            % 根据你原来的问题自行替换
            A = randn(n, r);
            b = randn(n, 1);
            % 例如：min_x 0.5*||Ax - b||^2 + lambda*||x||_1 之类
            % 这里只是示意：你可以根据自己的模型改成需要的数据结构

            % 初始点（按你原来的设定来）
            V0 = randn(m, r + 1);  
            % x0 = zeros(n, 1);

            % ====== 调用 DCRA ======
            t_start = tic;
            % ---- 下面这行替换成你真正的 DCRA 调用 ----

            [xsol_dcra, info_dcra] = DCRA(A, b, V0, @env_l1);
            % [x_dcra, info_dcra] = DCRA_solver(A, b, x0, opts_dcra);
            % 为了示例，这里随便造一个 info 结构：
            % info_dcra.obj  = rand();         % 目标值
            % info_dcra.res  = rand();         % 残差
            % info_dcra.iter = randi([5, 50]); % 迭代次数
            time_dcra      = toc(t_start);

            % ====== 调用 new_algo ======
            t_start = tic;
            % ---- 下面这行替换成你真正的 new_algo 调用 ----
            [x_new, info_new] = DCRA_intpl(A, b, V0, @env_l1);
            % [x_new, info_new] = new_algo_solver(A, b, x0, opts_new);
            % 同样只是示意：
            % info_new.obj  = rand();
            % info_new.res  = rand();
            % info_new.iter = randi([5, 50]);
            time_new      = toc(t_start);

            % -------- 累积结果 --------
            obj_dcra_all(itest)  = info_dcra.fobj;
            obj_new_all(itest)   = info_new.fobj;
            res_dcra_all(itest)  = info_dcra.res;
            res_new_all(itest)   = info_new.res;
            iter_dcra_all(itest) = info_dcra.itr;
            iter_new_all(itest)  = info_new.itr;
            time_dcra_all(itest) = time_dcra;
            time_new_all(itest)  = time_new;
        end

        % -------- 对 notest 次结果做平均 --------
        avg_obj_dcra  = mean(obj_dcra_all);
        avg_obj_new   = mean(obj_new_all);
        avg_res_dcra  = mean(res_dcra_all);
        avg_res_new   = mean(res_new_all);
        avg_iter_dcra = mean(iter_dcra_all);
        avg_iter_new  = mean(iter_new_all);
        avg_time_dcra = mean(time_dcra_all);
        avg_time_new  = mean(time_new_all);

        % -------- 填到 result_table --------
        result_table{row_idx, 1}  = r;
        result_table{row_idx, 2}  = n;
        result_table{row_idx, 3}  = avg_obj_dcra;
        result_table{row_idx, 4}  = avg_obj_new;
        result_table{row_idx, 5}  = avg_time_dcra;
        result_table{row_idx, 6} = avg_time_new;
        % result_table{row_idx, 7}  = avg_res_dcra;
        % result_table{row_idx, 8}  = avg_res_new;
        % result_table{row_idx, 9}  = avg_iter_dcra;
        % result_table{row_idx, 10}  = avg_iter_new;

        row_idx = row_idx + 1;
    end
end

% ---------------- 输出结果 ----------------
disp('Test Results:');
disp(result_table);

% save('result2.mat','result_table');
% writecell(result_table, 'result_table2.xlsx');
