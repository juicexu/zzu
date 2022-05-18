clear
close  all
%% ���ݶ�ȡ
%��Ҫ˵����һ�����ݵĲ��
shurugeshu=10 ;%ѡ��ǰ������Ϊ���ݵ�����%%%%����������������������������������������
%������������Ϊ��ǰ����ǰshurugeshu�������ݣ�
%���������Ԫ��shurugeshu������ǰ������Ϊ�������
lab=1330;%ѡ��ǰlab����Ϊѵ����,%%%%%%%%%%%%%%%%%%%%����������������
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
A=xlsread('ԭʼ����.xlsx');
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
%��������������ݹ�һ��

[aa,bb]=mapminmax([input_train input_test]);
[cc,dd]=mapminmax([output_train output_test]);
[inputn,inputps]=mapminmax('apply',input_train,bb);
[outputn,outputps]=mapminmax('apply',output_train,dd);

%% ģ�ͽ�����ѵ��

shuru_num = size(input_train,1); % ����ά��
shuchu_num = 1;  % ���ά��
zhongjian1_num = round(shuru_num); % LSTM�м��ڵ���
 
layers = [ ...
    sequenceInputLayer(shuru_num)
    lstmLayer(zhongjian1_num)
    reluLayer()
    fullyConnectedLayer(shuchu_num)
    regressionLayer];
 
options = trainingOptions('adam', ...  % �ݶ��½�
    'MaxEpochs',100, ...                % ����������
    'InitialLearnRate',0.01, ...      % ��ʼѧϰ��
    'Verbose',0, ...
    'Plots','training-progress');
% ѵ��LSTM
net = trainNetwork(inputn,outputn,layers,options);
%% Ԥ��
net = resetState(net);% ����ĸ���״̬���ܶԷ�������˸���Ӱ�졣��������״̬���ٴ�Ԥ�����С�
[~,Ytrain]= predictAndUpdateState(net,inputn);
test_simu=mapminmax('reverse',Ytrain,dd);%����һ��
%���Լ���������������ݹ�һ��
inputn_test=mapminmax('apply',input_test,bb);
[~,an]= predictAndUpdateState(net,inputn_test);
test_simu1=mapminmax('reverse',an,dd);%����һ��
%% ��ͼ
%��Ԥ��ֵ��������ݽ��бȽϡ�
figure
plot(output_train)
hold on
plot(test_simu,'.-')
hold off
% legend(['��ʵֵ','Ԥ��ֵ'])
xlabel('����')
title('ѵ����')
figure
plot(output_test)
hold on
plot(test_simu1,'.-')
hold off
% legend(['��ʵֵ','Ԥ��ֵ'])
xlabel('����')
title('���Լ�')
%% ����Ԥ��δ������

yuce_T = 10;  % ����Ԥ������
for i=1:yuce_T
a=i;
input_test=A((end-shurugeshu+1):end);
inputn_test=mapminmax('apply',input_test,inputps);
[~,an]= predictAndUpdateState(net,inputn_test);
test_simu1=mapminmax('reverse',an,outputps);
output(a)=test_simu1;
A=[A;test_simu1];
end
disp('Ԥ������')
disp(output)


