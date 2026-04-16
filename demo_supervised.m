clc; clear; close all;

% startup;
% 设置数据集和特征类型
str_dataset = 'MNIST'; % 'CIFAR10'  'USPS'  'NUSWIDE'  'iMDb'  'MIRFLICKR0'
% 'flickr' '256feat2048Norml'  'Caltech256-CNN1024dNorml'  'iapr-tc12' 'MNIST'
% 'MIRFLICKR25K'
feature = 'RBF'; % 'raw'  'RBF'

% 算法数量
usetemp = 0;
if usetemp 
    load tempdataset;
else
    % 加载数据集
    [traindata, traingnd, testdata, testgnd] = load_dataset(str_dataset);
    % shuffle操作 
    n = size(traindata,1); % 获取向量长度 
    n_t = size(testdata,1);
    % 生成随机排列的索引
    shuffle_idx = randperm(n);
    shuffle_idx_t = randperm(n_t);
    
    % 对A和B应用相同的随机排列
    traindata = traindata(shuffle_idx,:);
    traingnd = traingnd(shuffle_idx,:);
    testgnd = testgnd(shuffle_idx_t,:);
    testdata = testdata(shuffle_idx_t,:);
    
    
    % 数据处理
    [feaTrain, feaTest, Y, n, cateTrainTest] = data_processing(traindata, traingnd, testdata, testgnd, feature);
    clear traindata testdata;
end


% 设置随机种子
k_val = 20; 

opt.mitr = 5;
opt.mitr_inner = 10;
opt.tol = 1e-4;
opt.delta = .5;
opt.lambda = 0.25;
nalgo = 5;%4; 

loopnbits = [16 32 48 64 96]; % 64 96];%
pos = [1:10:40 50:50:1000];

% 初始化结果矩阵
cputime = zeros(nalgo, length(loopnbits)); 
Obj = zeros(nalgo, length(loopnbits));
mAP = cell(1, length(loopnbits), nalgo);
recall = cell(1, length(loopnbits), nalgo);
precision = cell(1, length(loopnbits), nalgo);
rec = cell(1, length(loopnbits), nalgo);
pre = cell(1, length(loopnbits), nalgo);

% 选择比特数
result_name = ['./results/' str_dataset '_' feature '_result.mat'];
try
load(result_name);
catch 
    fprintf('No previous results.\n')
end
nalgo = 5;%4;

% 算法名称
hashmethods ={'DCRA', 'MPEC-epm', 'L_2-box ADMM','Subgradient'}; % ,'Gurobi'
runtimes = 1;

%% 主循环
for k = 1:runtimes
    for ii = 1:length(loopnbits)
        nbits = loopnbits(ii); 
        fprintf('======start %d bits encoding======\n', nbits);
 
        for j = 1:nalgo %1%[1,8]%
            if j == 5
                continue
            end
            % 计时开始
            start_time = tic;

            % 选择不同的算法进行计算 
%             if j <= 3
            [H, Wg] = alt_min2(Y', nbits, j, opt); 
%             elseif j == 4
%                 [H, Wg] = alt_min(Y', nbits, 1, opt); 
%             elseif j == 5
%                 [H, Wg] = alt_min2(Y', nbits, 4, opt); 
%             end
%             [H, Wg] = DPLM_L3(Y, nbits, opt); 
            W = RRC(feaTrain, H, 1e-2);  

            Hz = sign(feaTrain * W);
            tH = sign(feaTest * W); 

            % 记录算法运行时间
            cputime(j, ii) = toc(start_time);

            % 计算汉明距离
            hammTrainTest = 0.5 * (nbits - Hz * tH');

            % 排序得到汉明距离排名
            [~, HammingRank] = sort(hammTrainTest, 1);

            % 计算召回率、精确率和平均精度均值
            [recall{k}{ii, j}, precision{k}{ii, j}, ~] = recall_precision(cateTrainTest', hammTrainTest');
	        [rec{k}{ii, j}, pre{k}{ii, j}]= recall_precision5(cateTrainTest', hammTrainTest', pos); % recall VS. the number of retrieved sample
            [mAP{k}{ii, j}] = area_RP(recall{k}{ii, j}, precision{k}{ii, j});
            %[mAP{k}{ii, j}] = sum(sum(abs(H*Wg - nbits*Y))) + opt.delta*norm(Wg,2)^2; %
            [~, ndcg{k}{ii, j}] = NDCG_test(traingnd, testgnd, HammingRank, 20, k_val);

            % 计算其他评估指标
        end
    end
end

% 计算平均mAP
MAP = zeros(length(loopnbits), nalgo);
for j = 1:nalgo
    for i = 1:length(loopnbits)
        sum_mAP = 0;
        for k = 1:runtimes
            sum_mAP = sum_mAP + mAP{k}{i, j};
        end
        MAP(i, j) = sum_mAP / runtimes;
    end
end

% 保存结果
% current_time = char(datetime('now', 'Format', 'yyyy-MM-dd_HH-mm-ss'));
result_name = ['./results/' str_dataset '_' feature '_result.mat'];
% save(result_name, 'precision', 'recall', 'rec', 'pre', 'MAP', 'mAP', 'hashmethods', ...
%     'nalgo', 'loopnbits', 'cputime',  'Obj','ndcg');

%% 绘图设置
line_width = 2;
marker_size = 8;
xy_font_size = 14;
legend_font_size = 12;
linewidth = 1.6;
title_font_size = xy_font_size;
choose_bits = 2;
choose_times = 1;

% 绘制召回率 vs 检索样本数的图
figure('Color', [1 1 1]);
hold on;
for j = 1:nalgo
    recc = rec{choose_times}{choose_bits, j};
    plot(pos, recc, 'Color', gen_color(j), 'Marker', gen_marker(j), ...
        'LineWidth', line_width, 'MarkerSize', marker_size);
end
str_nbits = num2str(loopnbits(choose_bits));
xlabel('The number of retrieved samples', 'FontSize', xy_font_size);
ylabel(['Recall @ ', str_nbits, ' bits'], 'FontSize', xy_font_size);
title(str_dataset, 'FontSize', title_font_size);
axis square;
legend(hashmethods, 'FontSize', legend_font_size, 'Location', 'best');
box on;
grid on;
hold off;

% 绘制精确率 vs 检索样本数的图
figure('Color', [1 1 1]);
hold on;
for j = 1:nalgo
    prec = pre{choose_times}{choose_bits, j};
    plot(pos, prec, 'Color', gen_color(j), 'Marker', gen_marker(j), ...
        'LineWidth', line_width, 'MarkerSize', marker_size);
end
xlabel('The number of retrieved samples', 'FontSize', xy_font_size);
ylabel(['Precision @ ', str_nbits, ' bits'], 'FontSize', xy_font_size);
title(str_dataset, 'FontSize', title_font_size);
axis square;
legend(hashmethods, 'FontSize', legend_font_size, 'Location', 'best');
box on;
grid on;
hold off;

% 绘制精确率 vs 召回率的图
figure('Color', [1 1 1]);
hold on;
for j = 1:nalgo
    plot(recall{choose_times}{choose_bits, j}, precision{choose_times}{choose_bits, j}, ...
        'Color', gen_color(j), 'Marker', gen_marker(j), ...
        'LineWidth', line_width, 'MarkerSize', marker_size);
end
xlabel(['Recall @ ', str_nbits, ' bits'], 'FontSize', xy_font_size);
ylabel('Precision', 'FontSize', xy_font_size);
title(str_dataset, 'FontSize', title_font_size);
axis square;
legend(hashmethods, 'FontSize', legend_font_size, 'Location', 'best');
box on;
grid on;
hold off;

% 绘制平均精度均值 vs 比特数的图
figure('Color', [1 1 1]);
hold on;
for j = 1:nalgo
    plot(log2(loopnbits), MAP(:, j), 'Color', gen_color(j), 'Marker', gen_marker(j), ...
        'LineWidth', line_width, 'MarkerSize', marker_size);
end
xlabel('Number of bits', 'FontSize', xy_font_size);
ylabel('mean Average Precision (mAP)', 'FontSize', xy_font_size);
title(str_dataset, 'FontSize', title_font_size);
axis square;
set(gca, 'xtick', log2(loopnbits));
set(gca, 'XtickLabel', loopnbits);
set(gca, 'linewidth', linewidth);
legend(hashmethods, 'FontSize', legend_font_size, 'Location', 'best');
box on;
grid on;
hold off;