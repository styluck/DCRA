function [feaTrain, feaTest, Y,n, cateTrainTest ] = data_processing(traindata, traingnd, testdata, testgnd, feature)
    % 将数据和标签转换为双精度类型
    traingnd = double(traingnd);
    traindata = double(traindata);
    testgnd = double(testgnd);
    testdata = double(testdata);

    % 训练集大小
    n = size(traindata, 1);

    % 计算训练集和测试集标签的对应关系
    cateTrainTest = bsxfun(@eq, traingnd, testgnd');

    % 根据不同的特征类型进行处理
    switch feature
        case 'raw'
            % 计算训练数据的均值
            m = mean(traindata);
            % 训练数据和测试数据减去均值
            feaTrain = bsxfun(@minus, traindata, m);
            feaTest = bsxfun(@minus, testdata, m);
        case 'ctr'
            % 初始化训练和测试特征矩阵
            feaTrain = zeros(size(traindata));
            feaTest = zeros(size(testdata));

            % 对训练数据进行归一化处理
            for i = 1:size(traindata, 1)
                row = traindata(i, :);
                min_val = min(row);
                max_val = max(row);
                if max_val - min_val > 0
                    projected_row = (row - min_val) / (max_val - min_val);
                else
                    projected_row = zeros(size(row));
                end
                feaTrain(i, :) = projected_row;
            end

            % 对测试数据进行归一化处理
            for i = 1:size(testdata, 1)
                row = testdata(i, :);
                min_val = min(row);
                max_val = max(row);
                if max_val - min_val > 0
                    projected_row = (row - min_val) / (max_val - min_val);
                else
                    projected_row = zeros(size(row));
                end
                feaTest(i, :) = projected_row;
            end
        case 'RBF'
            % 选取锚点
            n_anchors = 1000;
            rand('seed', 100);
            anchor = traindata(randperm(n, n_anchors), :);

            % 计算距离并确定sigma值
            Dis = EuDist2(traindata, anchor, 0);
            sigma = mean(min(Dis, [], 2).^0.5);
            clear Dis

            % 计算训练和测试特征
            feaTrain = exp(-sqdist(traindata, anchor) / (2 * sigma * sigma));
            feaTest = exp(-sqdist(testdata, anchor) / (2 * sigma * sigma));

            % 对特征进行中心化处理
            m = mean(feaTrain);
            feaTrain = bsxfun(@minus, feaTrain, m);
            feaTest = bsxfun(@minus, feaTest, m);

            % 计算亲和矩阵
%             [Z, lmb] = AffinityMatrix(traindata, anchor, 3, 0);
        otherwise
            error('未知的特征类型: %s', feature);
    end

    % 处理训练标签
    if isvector(traingnd)
        if nnz(traingnd) < length(traingnd)
            traingnd = traingnd + 1;
        end
        Y = sparse(1:length(traingnd), traingnd, 1);
        Y = full(Y);
    else
        Y = traingnd;
    end

    % 清除不再使用的变量
    clear traindata testdata
end
 % [EOF]