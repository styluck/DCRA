function [k_values, average_ndcg] = NDCG_test(traingnd, testgnd, HammingRank, n, k)

% 获取矩阵的行数
num_images = length(testgnd);
rows = size(HammingRank', 1);

% 初始化一个矩阵来存储每行最小的前 n 个数的索引
predicted_scores = zeros(rows, n);

for i = 1:rows
    % 获取当前行的数据
    sorted_indices = HammingRank(:, i)';
    % 提取前 n 个最小数的索引
    predicted_scores(i, :) = sorted_indices(1:n);
    predicted_scores(i, :) = traingnd(predicted_scores(i, :));
end

% 随机生成真实标签，实际使用时替换为真实的标签
% true_labels = randi([1, num_classes], num_images, 1);
true_labels = testgnd;
% 定义要计算 NDCG 的 k 值
k_values = 1:k;
num_k = length(k_values);
ndcg_per_image = zeros(num_images, num_k);

% 遍历每张图像计算 NDCG
for img_idx = 1:num_images
    % 获取当前图像的预测得分
    sorted_relevance = predicted_scores(img_idx, :) == true_labels(img_idx);

    % 计算不同 k 值下的 NDCG
    for k_idx = 1:num_k
        k = k_values(k_idx);
        % 计算 DCG
        dcg = 0;
        for j = 1:k
            dcg = dcg + (2^sorted_relevance(j) - 1) / log2(j + 1);
        end

        % 计算 IDCG
        ideal_labels = false(1, n);
        ideal_labels(true_labels(img_idx)) = true;
        [~, ideal_sorted_indices] = sort(double(ideal_labels), 'descend');
        ideal_sorted_relevance = ideal_labels(ideal_sorted_indices);
        idcg = 0;
        for j = 1:k
            idcg = idcg + (2^ideal_sorted_relevance(j) - 1) / log2(j + 1);
        end

        % 计算 NDCG
        if idcg > 0
            ndcg_per_image(img_idx, k_idx) = dcg / idcg;
        else
            ndcg_per_image(img_idx, k_idx) = 0;
        end
    end
end

% 计算所有图像的平均 NDCG
average_ndcg = mean(ndcg_per_image);

% 表格展示
% disp('k    Average NDCG');
% for k_idx = 1:num_k
%     fprintf('%d    %.4f\n', k_values(k_idx), average_ndcg(k_idx));
% end