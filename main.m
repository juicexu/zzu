clear
close  all
%% 数据读取
%重要说明，一个数据的拆分
shurugeshu=10 ;%选择前几天作为数据的输入%%%%！！！！！！！！！！！！！！！！！！！！
%首先输入样本为当前样本前shurugeshu个的数据，
%即输入层神经元有shurugeshu个，当前样本作为网络输出
lab=1330;%选择前lab个作为训练集,%%%%%%%%%%%%%%%%%%%%！！！！！！！！
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
A=xlsread('原始数据.xlsx');
k=1;
% [m,n] = 1490
for i=(shurugeshu+1):1:lab
input_train(k,1:shurugeshu)=A((i-shurugeshu):(i-1));
k=k+1;
end
input_train=input_train';
output_train=A((shurugeshu+1):1:(lab));
output_train=output_train';
k=1;
for i=(lab+1):1:size(A)
input_test(k,1:shurugeshu)=A((i-shurugeshu):(i-1));
k=k+1;
end
input_test=input_test';
output_test=A((lab+1):1:size(A));
output_test=output_test';
%样本输入输出数据归一化

[aa,bb]=mapminmax([input_train input_test]);
[cc,dd]=mapminmax([output_train output_test]);
[inputn,inputps]=mapminmax('apply',input_train,bb);
[outputn,outputps]=mapminmax('apply',output_train,dd);

%% 模型建立与训练

shuru_num = size(input_train,1); % 输入维度
shuchu_num = 1;  % 输出维度
zhongjian1_num = round(shuru_num); % LSTM中间层节点数
 
layers = [ ...
    sequenceInputLayer(shuru_num)
    lstmLayer(zhongjian1_num)
    reluLayer()
    fullyConnectedLayer(shuchu_num)
    regressionLayer];
 
options = trainingOptions('adam', ...  % 梯度下降
    'MaxEpochs',100, ...                % 最大迭代次数
    'InitialLearnRate',0.01, ...      % 初始学习率
    'Verbose',0, ...
    'Plots','training-progress');
% 训练LSTM
net = trainNetwork(inputn,outputn,layers,options);
%% 预测
net = resetState(net);% 网络的更新状态可能对分类产生了负面影响。重置网络状态并再次预测序列。
[~,Ytrain]= predictAndUpdateState(net,inputn);
test_simu=mapminmax('reverse',Ytrain,dd);%反归一化
%测试集样本输入输出数据归一化
inputn_test=mapminmax('apply',input_test,bb);
[~,an]= predictAndUpdateState(net,inputn_test);
test_simu1=mapminmax('reverse',an,dd);%反归一化
%% 画图
%将预测值与测试数据进行比较。
figure
plot(output_train)
hold on
plot(test_simu,'.-')
hold off
% legend(['真实值','预测值'])
xlabel('样本')
title('训练集')
figure
plot(output_test)
hold on
plot(test_simu1,'.-')
hold off
% legend(['真实值','预测值'])
xlabel('样本')
title('测试集')
%% 滚动预测未来数据

yuce_T = 10;  % 滚动预测天数
for i=1:yuce_T
a=i;
input_test=A((end-shurugeshu+1):end);
inputn_test=mapminmax('apply',input_test,inputps);
[~,an]= predictAndUpdateState(net,inputn_test);
test_simu1=mapminmax('reverse',an,outputps);
output(a)=test_simu1;
A=[A;test_simu1];
end
disp('预测数据')
disp(output)


