% -*- coding: GBK -*-
% Created on Sat May 14 2022 by Jiaqiang Liao
% 数据预处理及筛选模型，为N2O_04_Final_anlysis做准备
%% 1.数据导入
clc,clear all
% Ctrl+R 批量注释
% Ctrl+T 去除注释

% N2O data read  
datapath = 'D:\研究生学习\氮循环\N map\data\';
input = strcat(datapath,'Field nitrous oxide emission_by Lizhaolei.csv');

N2O_u = xlsread(input);  %读取N2O原始数据库
N2O_u = fillmissing(N2O_u,'movmean',6016);%取均值替代缺失值

MAT = N2O_u(:,8); 
MAP = N2O_u(:,9);
% sand = N2O_u(:,12);
% silt = N2O_u(:,13);
% clay = N2O_u(:,14);
BD = N2O_u(:,15);
pH = N2O_u(:,16);
SOC = log(N2O_u(:,17));
TN = log(N2O_u(:,19));
TP = log(N2O_u(:,21));
MBC = log(N2O_u(:,22));
MBN = log(N2O_u(:,23));
NO3 = log(N2O_u(:,25));
NH4 = log(N2O_u(:,26));
SM = N2O_u(:,30);

N2O_x = [MAT,MAP,BD,pH,SOC,TN,TP,SM,MBC,MBN,NO3,NH4];
N2O_y = N2O_u(:,28);

% 取对数
N2O_y = log(N2O_y);

% 去缺失值
% N2O_X = fillmissing(N2O_x,"constant",0); 
% N2O_Y = fillmissing(N2O_y,'constant',0);

% 数据标准化
% N2O_X = mapminmax(N2O_X);
% N2O_Y = mapminmax(N2O_Y);

N2O = [N2O_X,N2O_Y];

%% 2.分训练集和测试集效果可视化（observe vs predict）

% 划分训练集和测试集
[ndata, D] = size(N2O_X);          %ndata样本数，D维数
R = randperm(ndata,round(0.2*ndata));                    %1到n这些数随机打乱得到的一个随机数字序列作为索引
N2O_X_test = N2O_X(R,:);    %以索引的前20%个数据点作为测试样本Xtest
N2O_Y_test = N2O_Y(R,:);
N2O_X(R,:) = [];
N2O_Y(R,:) = [];
N2O_X_train = N2O_X;           %剩下的数据作为训练样本Xtraining
N2O_Y_train = N2O_Y;

%% 2.1 Random Forest
[trainedModel,validationRMSE,validationR2] = trainRegressionModel_RF(N2O_X_train,N2O_Y_train);
% 训练集性能
prdeict_N2O_RF = trainedModel.predictFcn(N2O_X_train);
plotregression(N2O_Y_train,prdeict_N2O_RF)
xlabel('Observerd N2O'),ylabel('Predict N2O')
legend('Random Forest','train data')
hold off
saveas(gcf,'D:\研究生学习\氮循环\N map\data\0-N20数据分析\RF-1.jpg')

plot(N2O_Y_train,'r')
hold on
plot(prdeict_N2O_RF,'b')
legend('Observerd N2O','Predict N2O')   
title('Random Forest','train data')
hold off
saveas(gcf,'D:\研究生学习\氮循环\N map\data\0-N20数据分析\RF-2.jpg')

% 测试集性能
prdeict_N2O_RF_test = trainedModel.predictFcn(N2O_X_test);
plotregression(N2O_Y_test,prdeict_N2O_RF_test)
xlabel('Observerd N2O'),ylabel('Predict N2O')
legend('Random Forest','test data')
hold off
saveas(gcf,'D:\研究生学习\氮循环\N map\data\0-N20数据分析\RF-3.jpg')

plot(N2O_Y_test,'r')
hold on
plot(prdeict_N2O_RF_test,'b')
legend('Observerd N2O','Predict N2O')
title('Random Forest','test data')
hold off
saveas(gcf,'D:\研究生学习\氮循环\N map\data\0-N20数据分析\RF-4.jpg')


%% 2.2 Artificial Neural Network
[trainedModel,validationRMSE,validationR2] = trainRegressionModel_ANN(N2O_X_train,N2O_Y_train);
% 训练集性能
prdeict_N2O_ANN = trainedModel.predictFcn(N2O_X_train);
plotregression(N2O_Y_train,prdeict_N2O_ANN)
xlabel('Observerd N2O'),ylabel('Predict N2O')
legend('Artificial Neural Network','train data')
hold off
saveas(gcf,'D:\研究生学习\氮循环\N map\data\0-N20数据分析\ANN-1.jpg')

plot(N2O_Y_train,'r')
hold on
plot(prdeict_N2O_ANN,'b')
legend('Observerd N2O','Predict N2O')
title('Artificial Neural Network','train data')
hold off
saveas(gcf,'D:\研究生学习\氮循环\N map\data\0-N20数据分析\ANN-2.jpg')

% 测试集性能
prdeict_N2O_ANN_test = trainedModel.predictFcn(N2O_X_test);
plotregression(N2O_Y_test,prdeict_N2O_ANN_test)
xlabel('Observerd N2O'),ylabel('Predict N2O')
legend('Artificial Neural Network','test data')
hold off
saveas(gcf,'D:\研究生学习\氮循环\N map\data\0-N20数据分析\ANN-3.jpg')

plot(N2O_Y_test,'r')
hold on
plot(prdeict_N2O_ANN_test,'b')
legend('Observerd N2O','Predict N2O')
title('Artificial Neural Network','test data')
hold off
saveas(gcf,'D:\研究生学习\氮循环\N map\data\0-N20数据分析\ANN-4.jpg')

%% 2.3 Gaussian Process Regression
[trainedModel,validationRMSE,validationR2] = trainRegressionModel_GRP(N2O_X_train,N2O_Y_train);
% 训练集性能
prdeict_N2O_GRP = trainedModel.predictFcn(N2O_X_train);
plotregression(N2O_Y_train,prdeict_N2O_GRP)
xlabel('Observerd N2O'),ylabel('Predict N2O')
legend('Gaussian Process Regression','train data')
hold off
saveas(gcf,'D:\研究生学习\氮循环\N map\data\0-N20数据分析\GRP-1.jpg')

plot(N2O_Y_train,'r')
hold on
plot(prdeict_N2O_GRP,'b')
legend('Observerd N2O','Predict N2O')
title('Gaussian Process Regression','train data')
hold off
saveas(gcf,'D:\研究生学习\氮循环\N map\data\0-N20数据分析\GRP-2.jpg')

% 测试集性能
prdeict_N2O_GRP_test = trainedModel.predictFcn(N2O_X_test);
plotregression(N2O_Y_test,prdeict_N2O_GRP_test)
xlabel('Observerd N2O'),ylabel('Predict N2O')
legend('Gaussian Process Regression','test data')
hold off
saveas(gcf,'D:\研究生学习\氮循环\N map\data\0-N20数据分析\GRP-3.jpg')

plot(N2O_Y_test,'r')
hold on
plot(prdeict_N2O_GRP_test,'b')
legend('Observerd N2O','Predict N2O')
title('Gaussian Process Regression','test data')
hold off
saveas(gcf,'D:\研究生学习\氮循环\N map\data\0-N20数据分析\GRP-4.jpg')

%% 2.4 Decision Tree
[trainedModel,validationRMSE,validationR2] = trainRegressionModel_DT(N2O_X_train,N2O_Y_train);
% 训练集性能
prdeict_N2O_DT = trainedModel.predictFcn(N2O_X_train);
plotregression(N2O_Y_train,prdeict_N2O_DT)
xlabel('Observerd N2O'),ylabel('Predict N2O')
legend('Decision Tree','train data')
hold off
saveas(gcf,'D:\研究生学习\氮循环\N map\data\0-N20数据分析\DT-1.jpg')

plot(N2O_Y_train,'r')
hold on
plot(prdeict_N2O_DT,'b')
legend('Observerd N2O','Predict N2O')
title('Decision Tree','train data')
hold off
saveas(gcf,'D:\研究生学习\氮循环\N map\data\0-N20数据分析\DT-2.jpg')

% 测试集性能
prdeict_N2O_DT_test = trainedModel.predictFcn(N2O_X_test);
plotregression(N2O_Y_test,prdeict_N2O_DT_test)
xlabel('Observerd N2O'),ylabel('Predict N2O')
legend('Decision Tree','test data')
hold off
saveas(gcf,'D:\研究生学习\氮循环\N map\data\0-N20数据分析\DT-3.jpg')

plot(N2O_Y_test,'r')
hold on
plot(prdeict_N2O_DT_test,'b')
legend('Observerd N2O','Predict N2O')
title('Decision Tree','test data')
hold off
saveas(gcf,'D:\研究生学习\氮循环\N map\data\0-N20数据分析\DT-4.jpg')

%% 2.5 Support Vector Machine
[trainedModel,validationRMSE,validationR2] = trainRegressionModel_SVM(N2O_X_train,N2O_Y_train);
% 训练集性能
prdeict_N2O_SVM = trainedModel.predictFcn(N2O_X_train);
plotregression(N2O_Y_train,prdeict_N2O_SVM)
xlabel('Observerd N2O'),ylabel('Predict N2O')
legend('Support Vector Machine','train data')
hold off
saveas(gcf,'D:\研究生学习\氮循环\N map\data\0-N20数据分析\SVM-1.jpg')


plot(N2O_Y_train,'r')
hold on
plot(prdeict_N2O_SVM,'b')
legend('Observerd N2O','Predict N2O')
title('Support Vector Machine','train data')
hold off
saveas(gcf,'D:\研究生学习\氮循环\N map\data\0-N20数据分析\SVM-2.jpg')

% 测试集性能
prdeict_N2O_SVM_test = trainedModel.predictFcn(N2O_X_test);
plotregression(N2O_Y_test,prdeict_N2O_SVM_test)
xlabel('Observerd N2O'),ylabel('Predict N2O')
legend('Support Vector Machine','test data')
hold off
saveas(gcf,'D:\研究生学习\氮循环\N map\data\0-N20数据分析\SVM-3.jpg')

plot(N2O_Y_test,'r')
hold on
plot(prdeict_N2O_SVM_test,'b')
legend('Observerd N2O','Predict N2O')
title('Support Vector Machine','test data')
hold off
saveas(gcf,'D:\研究生学习\氮循环\N map\data\0-N20数据分析\SVM-4.jpg')

%% 2.6 Stepwise Linear Regression
[trainedModel,validationRMSE,validationR2] = trainRegressionModel_stepwiselm(N2O_X_train,N2O_Y_train);
% 训练集性能
prdeict_N2O_step = trainedModel.predictFcn(N2O_X_train);
plotregression(N2O_Y_train,prdeict_N2O_step)
xlabel('Observerd N2O'),ylabel('Predict N2O')
legend('Stepwise Linear Regression','train data')
hold off
saveas(gcf,'D:\研究生学习\氮循环\N map\data\0-N20数据分析\step-1.jpg')

plot(N2O_Y_train,'r')
hold on
plot(prdeict_N2O_step,'b')
legend('Observerd N2O','Predict N2O')
title('Stepwise Linear Regression','train data')
hold off
saveas(gcf,'D:\研究生学习\氮循环\N map\data\0-N20数据分析\step-2.jpg')

% 测试集性能
prdeict_N2O_step_test = trainedModel.predictFcn(N2O_X_test);
plotregression(N2O_Y_test,prdeict_N2O_step_test)
xlabel('Observerd N2O'),ylabel('Predict N2O')
legend('Stepwise Linear Regression','test data')
hold off
saveas(gcf,'D:\研究生学习\氮循环\N map\data\0-N20数据分析\step-3.jpg')

plot(N2O_Y_test,'r')
hold on
plot(prdeict_N2O_step_test,'b')
legend('Observerd N2O','Predict N2O')
title('Stepwise Linear Regression','test data')
hold off
saveas(gcf,'D:\研究生学习\氮循环\N map\data\0-N20数据分析\step-4.jpg')

%% 3.所有数据进行训练六种模型并进行交叉验证看总体效果并预测格局

% 数据量(n = 6016)
% 10折交叉验证

% Random Forest
for i = 1:10
[trainedModel,validationRMSE(i),validationR2(i)] = trainRegressionModel_RF(N2O_X,N2O_Y);
end
RF_RMSE = validationRMSE
RF_R2 = validationR2


% Artificial Neural Network
for i = 1:10
[trainedModel,validationRMSE(i),validationR2(i)] = trainRegressionModel_ANN(N2O_X,N2O_Y);
end
ANN_RMSE = validationRMSE
ANN_R2 = validationR2

% Gaussian Process Regression
for i = 1:10
[trainedModel,validationRMSE(i),validationR2(i)] = trainRegressionModel_GRP(N2O_X,N2O_Y);
end
GRP_RMSE = validationRMSE
GRP_R2 = validationR2

% Decision Tree
for i = 1:10
[trainedModel,validationRMSE(i),validationR2(i)] = trainRegressionModel_DT(N2O_X,N2O_Y);
end
DT_RMSE = validationRMSE
DT_R2 = validationR2

% Support Vector Machine
for i = 1:10
[trainedModel,validationRMSE(i),validationR2(i)] = trainRegressionModel_SVM(N2O_X,N2O_Y);
end
SVM_RMSE = validationRMSE
SVM_R2 = validationR2

% Stepwise Linear Regression
% for i = 1:10
% [trainedModel,validationRMSE(i),validationR2(i)] = trainRegressionModel_stepwiselm(N2O_X,N2O_Y);
% end
% Step_RMSE = validationRMSE
% Step_R2 = validationR2

model_result = [GRP_R2',GRP_RMSE',ANN_R2',ANN_RMSE',RF_R2',RF_RMSE',DT_R2',DT_RMSE',SVM_R2',SVM_RMSE',Step_R2',Step_RMSE'];

