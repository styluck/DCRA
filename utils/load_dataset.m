function [traindata, traingnd, testdata, testgnd] = load_dataset(str_dataset)
% folder_path = 'E:\Users\Documents\OneDrive - jnu.edu.cn\Documents\Projects\hashing\codes\paper_codes\dataset';
folder_path = 'E:\Documents\OneDrive - jnu.edu.cn\Documents\Projects\hashing\codes\paper_codes\dataset';
    % 根据数据集名称载入相应的数据
    switch str_dataset
        case 'MNIST'
            % 载入MNIST数据集
            file_path = fullfile(folder_path, 'mnisttraindata.mat');
            load(file_path);
            file_path = fullfile(folder_path, 'mnisttestdata.mat'); 
            load(file_path);

        case 'CIFAR10'
            % 载入CIFAR10数据集
            file_path = fullfile(folder_path, 'cifartestdata.mat');
            load(file_path);
            file_path = fullfile(folder_path, 'cifartraindata.mat'); 
            load(file_path); 
            traingnd = traingnd + 1;
            testgnd = testgnd + 1;
        case 'USPS'
            % 载入USPS数据集
            file_path = fullfile(folder_path, 'USPStest.mat');
            load(file_path);
            file_path = fullfile(folder_path, 'USPStrain.mat'); 
            load(file_path); 
%             load USPStest.mat;
%             load USPStrain.mat;
        case 'NUSWIDE'
            % 载入NUSWIDE数据集
            file_path = fullfile(folder_path, 'NUSWIDEtestdata.mat');
            load(file_path);
            file_path = fullfile(folder_path, 'NUSWIDEtraindata.mat'); 
            load(file_path); 
%             load NUSWIDEtestdata.mat;
%             load NUSWIDEtraindata.mat;
            
            % 处理NUSWIDE数据集
            % 统计训练标签中每个样本的类别数量
            cnum = sum(traingnd, 2);
            % 找出只有一个类别的样本索引
            idx = find(cnum == 1);
            li = length(idx);
            % 初始化新的训练标签
            traingnd1 = zeros(li, 1);
            % 提取只有一个类别的训练标签
            for i = 1:li
                traingnd1(i) = find(traingnd(idx(i), :));
            end
            % 提取只有一个类别的训练数据
            traindata = traindata(idx, :);

            % 统计测试标签中每个样本的类别数量
            cnum = sum(testgnd, 2);
            % 找出只有一个类别的样本索引
            idx = find(cnum == 1);
            li = length(idx);
            % 初始化新的测试标签
            testgnd1 = zeros(li, 1);
            % 提取只有一个类别的测试标签
            for i = 1:li
                testgnd1(i) = find(testgnd(idx(i), :));
            end
            % 提取只有一个类别的测试数据
            testdata = testdata(idx, :);

            % 更新训练和测试标签
            traingnd = traingnd1;
            testgnd = testgnd1;

            % 清除临时变量
            clear traingnd1 testgnd1;
            
        case 'iMDb'
            % 处理iMDb数据集  
            file_path = fullfile(folder_path, 'imdb2.mat'); 
            load(file_path); 
%             load imdb2.mat;
            
        case 'flickr'  
            file_path = fullfile(folder_path, 'flickr.mat'); 
            load(file_path); 
%             load flickr.mat; 

            % 统计测试标签中每个样本的类别数量
            cnum = sum(YTrain, 2);
            % 找出只有一个类别的样本索引
            idx = find(cnum == 1);
            li = length(idx);
            % 初始化新的测试标签
            traingnd = zeros(li, 1);
            % 提取只有一个类别的测试标签
            for i = 1:li
                traingnd(i) = find(YTrain(idx(i), :));
            end
            traindata = XTrain(idx, :);

            cnum = sum(YTest, 2);
            % 找出只有一个类别的样本索引
            idx = find(cnum == 1);
            li = length(idx);
            % 初始化新的测试标签
            testgnd = zeros(li, 1);
            % 提取只有一个类别的测试标签
            for i = 1:li
                testgnd(i) = find(YTest(idx(i), :));
            end
            % 提取只有一个类别的测试数据
            testdata = XTest(idx, :);
        case '256feat2048Norml' 
            file_path = fullfile(folder_path, '256feat2048Norml.mat'); 
            load(file_path); 
%             load 256feat2048Norml.mat; 
            n = length(rgbImgList);
            ntest = round(n/10);
            traingnd = zeros(n, 1);
            
            % 遍历cell数组
            for i = 1:numel(rgbImgList)
                % 提取前3个字符
                traingnd(i) = str2double(rgbImgList{i}(1:3));
            end
            randomIndices = randperm(n);
            traindata = feat(randomIndices,:);
            traingnd = traingnd(randomIndices);

            testdata = traindata(1:ntest,:);
            testgnd = traingnd(1:ntest);
            traindata = traindata(ntest+1:end,:);
            traingnd = traingnd(ntest+1:end);

        case 'Caltech256-CNN1024dNorml'
            file_path = fullfile(folder_path, 'Caltech256-CNN1024dNorml.mat');
            load(file_path); 
%             load Caltech256-CNN1024dNorml.mat; 
            n = length(rgbImgList);
            ntest = round(n/10);
            traingnd = zeros(n, 1);
            
            % 遍历cell数组
            for i = 1:numel(rgbImgList)
                % 提取前3个字符
                traingnd(i) = str2double(rgbImgList{i}(1:3));
            end

            [traindata, traingnd, testdata, testgnd] = train_n_test_split(feat, traingnd,ntest);

        case 'iapr-tc12' 
            file_path = fullfile(folder_path, 'iapr-tc12.mat'); 
            load(file_path); 
%             load iapr-tc12.mat;
            % 统计测试标签中每个样本的类别数量
            [traindata, traingnd] = getlabel(XDatabase, YDatabase);
            [testdata, testgnd] = getlabel(XTest, YTest);
 
%             testdata = VTest(idx, :);
        case 'wiki'
            file_path = fullfile(folder_path, 'WikiData.mat');
            load(file_path); 
%             load WikiData.mat;
            [traindata, traingnd] = getlabel(I_tr, L_tr);
            [testdata, testgnd] = getlabel(I_te, L_te);
        case 'MIRFLICKR0'
            file_path = fullfile(folder_path, 'MIRFLICKR0.mat');
            load(file_path); 
%             load MIRFLICKR0.mat;
            [traindata, traingnd] = getlabel(T_tr, L_tr);
            [testdata, testgnd] = getlabel(T_te, L_te);
        case 'MIRFLICKR25K'
            file_path = fullfile(folder_path, 'MIRFLICKR25K.mat');
            load(file_path); 
%             load MIRFLICKR25K.mat;
            [traindata, traingnd] = getlabel(X_CNN, L);
            [traindata, traingnd, testdata, testgnd] = train_n_test_split(...
                traindata, traingnd, 500);

        otherwise
            error('未知的数据集名称: %s', str_dataset);
    end
end

function [x,y] = getlabel(X, Y)
    
    cnum = sum(Y, 2);
    % 找出只有一个类别的样本索引
    idx = find(cnum == 1);
    li = length(idx);
    % 初始化新的测试标签
    y = zeros(li, 1);
    % 提取只有一个类别的测试标签
    for i = 1:li
        y(i) = find(Y(idx(i), :));
    end
    % 提取只有一个类别的测试数据
    x = X(idx, :);
end

function [tainx, trainy, testx,testy] = train_n_test_split(X, Y, ntest)
    n = size(Y, 1);
    randomIndices = randperm(n);
    tainx = X(randomIndices,:);
    trainy = Y(randomIndices);

    testx = X(1:ntest,:);
    testy = Y(1:ntest);
    tainx = tainx(ntest+1:end,:);
    trainy = trainy(ntest+1:end);
end