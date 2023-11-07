% -*- coding: GBK -*-
% Created on Nov 2 2022 by Jiaqiang Liao
% N20站点数据upscale到全球尺度的主体分析过程，循环100次看结果
clc,clear all

% 加载数据
load N2O_database.mat % 载入训练数据集(6016)

%选取特定观测时长的数据集进行分析(N2O,N2O_1y,N2O_6m,N2O_1m)
[n, D] = size(N2O);
ID = 1:n;
rng(200)
samples = randsample(ID, n, true, weights); %权重取样，有放回
N2O = N2O(ID,:);

N2O_X = N2O(:,1:13);
N2O_Y = N2O(:,16);

%load Area_WGS_1984_720_360.mat
%% 1 固定随机种子构建单次模型
%不boot抽取，不随机划分训练测试数据集，固定一套数据的情况下，做单次拟合效果图和偏向关回归的结果
%% 1.1 forest（679）
N2O_X_forest = N2O_X(LandID == 3,:);
N2O_Y_forest = N2O_Y(LandID == 3,:);
Input = N2O_X_forest;
Output = N2O_Y_forest;

rng(200)  %设置随机种子
[ndata, D] = size(N2O_X_forest);          %ndata样本数，D维数
R = randperm(ndata,round(0.2*ndata));      %1到n这些数随机打乱得到的一个随机数字序列作为索引
N2O_X_test = N2O_X_forest(R,:);     %20%个数据点作为测试样本Xtest
N2O_Y_test = N2O_Y_forest(R,:);
N2O_X_forest(R,:) = [];
N2O_Y_forest(R,:) = [];
N2O_X_train = N2O_X_forest;           %剩下的数据作为训练样本Xtraining
N2O_Y_train = N2O_Y_forest;

% 模型训练
RFRunNumSet=10;
for RFCycleRun=1:RFRunNumSet
nTree=200;
nLeaf=5;
RFModel_forest=TreeBagger(nTree,N2O_X_train,N2O_Y_train,...
    'Method','regression','OOBPredictorImportance','on', 'MinLeafSize',nLeaf, ...
    'PredictorNames',{'MAT','MAP','BD','pH','SOC','TN','TP','SM','MBC','MBN','NO3','NH4','Nfer'});
[RFPredictYield,RFPredictConfidenceInterval]=predict(RFModel_forest,N2O_X_test);
% Accuracy of RF    
RFRMSE=sqrt(sum(sum((RFPredictYield-N2O_Y_test).^2))/size(N2O_Y_test,1));
if RFRMSE<1.16     %当RMSE满足<X条件时，模型将自动停止
    disp(RFRMSE);
    break;
end
end
forest_R2 = 1-(sumsqr(N2O_Y_test-RFPredictYield)/sumsqr(N2O_Y_test-mean(N2O_Y_test)))

%模型效果拟合图
plotregression(N2O_Y_test,RFPredictYield)

figure()
plot(log(exp(N2O_Y_test)*365*0.000001),log(exp(RFPredictYield)*365*0.000001),'.','MarkerSize',15);
axis([-6 2 -6 2]);
set(gcf,'position',[100,100,500,450]);
line([-6:2],[-6:2],'color','k')
xlabel('Observerd, ln(N_2O,g m^-^2 year^-^1)'),ylabel('Predict, ln(N_2O,g m^-^2 year^-^1)')
text(-5.5,1,'Fitting slope = 0.7204','FontSize',10)
text(-5.5,1.5,'Ecosystem type: forest','FontSize',10)
print(gcf,'Regression_forest','-dpng','-r600')

%变量记录
RFPredict_global = RFPredictYield;  %记录各生态系统数据，为global做准备
N2O_Y_test_global = N2O_Y_test;     %记录各生态系统数据，为global做准备
RI_forest = RFModel_forest.OOBPermutedPredictorDeltaError/max(RFModel_forest.OOBPermutedPredictorDeltaError);

%% 1.2 grassland(n=514)
N2O_X_grassland = N2O_X(LandID == 4,:);
N2O_Y_grassland = N2O_Y(LandID == 4,:);
Input = N2O_X_grassland;
Output = N2O_Y_grassland;

rng(200)
[ndata, D] = size(N2O_X_grassland);          %ndata样本数，D维数
R = randperm(ndata,round(0.2*ndata));      %1到n这些数随机打乱得到的一个随机数字序列作为索引
N2O_X_test = N2O_X_grassland(R,:);     %20%个数据点作为测试样本Xtest
N2O_Y_test = N2O_Y_grassland(R,:);
N2O_X_grassland(R,:) = [];
N2O_Y_grassland(R,:) = [];
N2O_X_train = N2O_X_grassland;           %剩下的数据作为训练样本Xtraining
N2O_Y_train = N2O_Y_grassland;

% 模型训练
RFRunNumSet=10;
for RFCycleRun=1:RFRunNumSet
nTree=100;
nLeaf=5;
RFModel_grassland=TreeBagger(nTree,N2O_X_train,N2O_Y_train,...
    'Method','regression','OOBPredictorImportance','on', 'MinLeafSize',nLeaf, ...
    'PredictorNames',{'MAT','MAP','BD','pH','SOC','TN','TP','SM','MBC','MBN','NO3','NH4','Nfer'});
[RFPredictYield,RFPredictConfidenceInterval]=predict(RFModel_grassland,N2O_X_test);
% Accuracy of RF    
RFRMSE=sqrt(sum(sum((RFPredictYield-N2O_Y_test).^2))/size(N2O_Y_test,1));
if RFRMSE<1.2    %当RMSE满足<X条件时，模型将自动停止
    disp(RFRMSE);
    break;
end
end
grassland_R2 = 1-(sumsqr(N2O_Y_test-RFPredictYield)/sumsqr(N2O_Y_test-mean(N2O_Y_test)))

%模型效果拟合图
plotregression(N2O_Y_test,RFPredictYield)

figure()
plot(log(exp(N2O_Y_test)*365*0.000001),log(exp(RFPredictYield)*365*0.000001),'.','MarkerSize',15);
axis([-6 2 -6 2]);
set(gcf,'position',[100,100,500,450]);
line([-6:2],[-6:2],'color','k')
xlabel('Observerd, ln(N_2O,g m^-^2 year^-^1)'),ylabel('Predict, ln(N_2O,g m^-^2 year^-^1)')
text(-5.5,1,'Fitting slope = 0.8307','FontSize',10)
text(-5.5,1.5,'Ecosystem type: grassland','FontSize',10)
print(gcf,'Regression_grassland','-dpng','-r600')

%变量记录
RFPredict_global = [RFPredict_global,
    RFPredictYield];  %记录各生态系统数据，为global做准备
N2O_Y_test_global = [N2O_Y_test_global,
    N2O_Y_test];     %记录各生态系统数据，为global做准备
RI_grassland = RFModel_grassland.OOBPermutedPredictorDeltaError/max(RFModel_grassland.OOBPermutedPredictorDeltaError);

%% 1.3 croplands/management (n=4356)
N2O_X_croplands = N2O_X(LandID == 1,:);
N2O_Y_croplands = N2O_Y(LandID == 1,:);
Input = N2O_X_croplands;
Output = N2O_Y_croplands;

rng(200)
[ndata, D] = size(N2O_X_croplands);          %ndata样本数，D维数
R = randperm(ndata,round(0.2*ndata));      %1到n这些数随机打乱得到的一个随机数字序列作为索引
N2O_X_test = N2O_X_croplands(R,:);     %20%个数据点作为测试样本Xtest
N2O_Y_test = N2O_Y_croplands(R,:);
N2O_X_croplands(R,:) = [];
N2O_Y_croplands(R,:) = [];
N2O_X_train = N2O_X_croplands;           %剩下的数据作为训练样本Xtraining
N2O_Y_train = N2O_Y_croplands;

% 模型训练
RFRunNumSet=10;
for RFCycleRun=1:RFRunNumSet
nTree=100;nLeaf=5;
RFModel_croplands=TreeBagger(nTree,N2O_X_train,N2O_Y_train,...
    'Method','regression','OOBPredictorImportance','on', 'MinLeafSize',nLeaf, ...
    'PredictorNames',{'MAT','MAP','BD','pH','SOC','TN','TP','SM','MBC','MBN','NO3','NH4','Nfer'});
[RFPredictYield,RFPredictConfidenceInterval]=predict(RFModel_croplands,N2O_X_test);

% Accuracy of RF    
RFRMSE=sqrt(sum(sum((RFPredictYield-N2O_Y_test).^2))/size(N2O_Y_test,1));
if RFRMSE<1.1      %当RMSE满足<X条件时，模型将自动停止
    disp(RFRMSE);
    break;
end
end
cropland_R2 = 1-(sumsqr(N2O_Y_test-RFPredictYield)/sumsqr(N2O_Y_test-mean(N2O_Y_test)))

%模型效果拟合图
plotregression(N2O_Y_test,RFPredictYield)

figure()
plot(log(exp(N2O_Y_test)*365*0.000001),log(exp(RFPredictYield)*365*0.000001),'.','MarkerSize',15);
axis([-6 2 -6 2]);
set(gcf,'position',[100,100,500,450]);
line([-6:2],[-6:2],'color','k')
xlabel('Observerd, ln(N_2O,g m^-^2 year^-^1)'),ylabel('Predict, ln(N_2O,g m^-^2 year^-^1)')
text(-5.5,1,'Fitting slope = 0.7954','FontSize',10)
text(-5.5,1.5,'Ecosystem type: cropland','FontSize',10)
print(gcf,'Regression_cropland','-dpng','-r600')

%变量记录
RFPredict_global = [RFPredict_global,
    RFPredictYield];  %记录各生态系统数据，为global做准备
N2O_Y_test_global = [N2O_Y_test_global,
    N2O_Y_test];     %记录各生态系统数据，为global做准备
RI_cropland = RFModel_croplands.OOBPermutedPredictorDeltaError/max(RFModel_croplands.OOBPermutedPredictorDeltaError);

%% 1.4 global总体表现
plotregression(N2O_Y_test_global,RFPredict_global)

figure()
plot(log(exp(N2O_Y_test_global)*365*0.000001),log(exp(RFPredict_global)*365*0.000001),'.','MarkerSize',15);
axis([-6 2 -6 2]);
set(gcf,'position',[100,100,500,450]);
line([-6:2],[-6:2],'color','k')
xlabel('Observerd, ln(N_2O,g m^-^2 year^-^1)'),ylabel('Predict, ln(N_2O,g m^-^2 year^-^1)')
text(-5.5,1,'Fitting slope = 0.7940','FontSize',10)
text(-5.5,1.5,'Ecosystem type: global','FontSize',10)
print(gcf,'Regression_global','-dpng','-r600')

% Accuracy of global performence
Global_RFR2 = 1-(sumsqr(N2O_Y_test_global-RFPredict_global)/sumsqr(N2O_Y_test_global-mean(N2O_Y_test_global)))

%% 1.5 偏向关分析图
subplot(3,5,1)
plotPartialDependence(RFModel_forest,1)
hold on
plotPartialDependence(RFModel_grassland,1)
plotPartialDependence(RFModel_croplands,1)
xlabel('MAT'),ylabel(''),title(' ')
ax = gca;set(findobj(get(gca,'Children'),'LineWidth',0.5),'LineWidth',2);
hold off

subplot(3,5,2)
plotPartialDependence(RFModel_forest,2)
hold on
plotPartialDependence(RFModel_grassland,2)
plotPartialDependence(RFModel_croplands,2)
xlabel('MAP'),ylabel(''),title(' ')
ax = gca;set(findobj(get(gca,'Children'),'LineWidth',0.5),'LineWidth',2);
hold off

subplot(3,5,3)
plotPartialDependence(RFModel_forest,3)
hold on
plotPartialDependence(RFModel_grassland,3)
plotPartialDependence(RFModel_croplands,3)
xlabel('BD'),ylabel(''),title(' ')
ax = gca;set(findobj(get(gca,'Children'),'LineWidth',0.5),'LineWidth',2);
hold off

subplot(3,5,4)
plotPartialDependence(RFModel_forest,4)
hold on
plotPartialDependence(RFModel_grassland,4)
plotPartialDependence(RFModel_croplands,4)
xlabel('pH'),ylabel(''),title(' ')
ax = gca;set(findobj(get(gca,'Children'),'LineWidth',0.5),'LineWidth',2);
hold off

subplot(3,5,5)
plotPartialDependence(RFModel_forest,5)
hold on
plotPartialDependence(RFModel_grassland,5)
plotPartialDependence(RFModel_croplands,5)
xlabel('SOC'),ylabel(''),title(' ')
ax = gca;set(findobj(get(gca,'Children'),'LineWidth',0.5),'LineWidth',2);
hold off

subplot(3,5,6)
plotPartialDependence(RFModel_forest,6)
hold on
plotPartialDependence(RFModel_grassland,6)
plotPartialDependence(RFModel_croplands,6)
xlabel('TN'),ylabel(''),title(' ')
ax = gca;set(findobj(get(gca,'Children'),'LineWidth',0.5),'LineWidth',2);
hold off

subplot(3,5,7)
plotPartialDependence(RFModel_forest,7)
hold on
plotPartialDependence(RFModel_grassland,7)
plotPartialDependence(RFModel_croplands,7)
xlabel('TP'),ylabel(''),title(' ')
ax = gca;set(findobj(get(gca,'Children'),'LineWidth',0.5),'LineWidth',2);
hold off

subplot(3,5,8)
plotPartialDependence(RFModel_forest,8)
hold on
plotPartialDependence(RFModel_grassland,8)
plotPartialDependence(RFModel_croplands,8)
xlabel('SWC'),ylabel(''),title(' ')
ax = gca;set(findobj(get(gca,'Children'),'LineWidth',0.5),'LineWidth',2);
hold off

subplot(3,5,9)
plotPartialDependence(RFModel_forest,9)
hold on
plotPartialDependence(RFModel_grassland,9)
plotPartialDependence(RFModel_croplands,9)
xlabel('MBC'),ylabel(''),title(' ')
ax = gca;set(findobj(get(gca,'Children'),'LineWidth',0.5),'LineWidth',2);
hold off

subplot(3,5,10)
plotPartialDependence(RFModel_forest,10)
hold on
plotPartialDependence(RFModel_grassland,10)
plotPartialDependence(RFModel_croplands,10)
xlabel('MBN'),ylabel(''),title(' ')
ax = gca;set(findobj(get(gca,'Children'),'LineWidth',0.5),'LineWidth',2);
hold off

subplot(3,5,11)
plotPartialDependence(RFModel_forest,11)
hold on
plotPartialDependence(RFModel_grassland,11)
plotPartialDependence(RFModel_croplands,11)
xlabel('NO3'),ylabel(''),title(' ')
ax = gca;set(findobj(get(gca,'Children'),'LineWidth',0.5),'LineWidth',2);
hold off

subplot(3,5,12)
plotPartialDependence(RFModel_forest,12)
hold on
plotPartialDependence(RFModel_grassland,12)
plotPartialDependence(RFModel_croplands,12)
xlabel('NH4'),ylabel(''),title(' ')
ax = gca;set(findobj(get(gca,'Children'),'LineWidth',0.5),'LineWidth',2);
hold off

subplot(3,5,13)
plotPartialDependence(RFModel_forest,13)
hold on
plotPartialDependence(RFModel_grassland,13)
plotPartialDependence(RFModel_croplands,13)
xlabel('Nfer'),ylabel(''),title(' ')
ax = gca;set(findobj(get(gca,'Children'),'LineWidth',0.5),'LineWidth',2);
hold off

print -djpeg -r600 N2O-Partial-dependce

%% 2 构建随机森林模型(循环100次)
clc,clear all

load N2O_database.mat 

for i = 1:100 %bootsrap算法100次,求mean和sd和CV；函数boot(X,n)
%按观测时长加权取样！！
[n, D] = size(N2O);
ID = 1:n;
samples = randsample(ID, n, true, weights); %权重取样，有放回
N2O = N2O(ID,:);

N2O_X = N2O(:,1:13);
N2O_Y = N2O(:,16);

%选取特定观测时长的数据集进行分析(N2O,N2O_1y,N2O_6m,N2O_1m)
% N2O_X = N2O_6m(:,1:12);
% N2O_Y = N2O_6m(:,16);
% Landcover = Landcover_6m;
% LandID = LandID_6m;

%% 2.1 forest
N2O_X_forest = N2O_X(LandID == 3,:);
N2O_Y_forest = N2O_Y(LandID == 3,:);
boot_id = (1:679); % 1y=568, 6m=622, 1m=662, total=679
boot_id = boot(boot_id,100);
Input = N2O_X_forest(boot_id(:,i),:);
Output = N2O_Y_forest(boot_id(:,i),:);
[ndata, D] = size(N2O_X_forest);          %ndata样本数，D维数
R = randperm(ndata,round(0.2*ndata));      %1到n这些数随机打乱得到的一个随机数字序列作为索引
N2O_X_test = N2O_X_forest(R,:);     %20%个数据点作为测试样本Xtest
N2O_Y_test = N2O_Y_forest(R,:);
N2O_X_forest(R,:) = [];
N2O_Y_forest(R,:) = [];
N2O_X_train = N2O_X_forest;           %剩下的数据作为训练样本Xtraining
N2O_Y_train = N2O_Y_forest;

% 模型训练
RFRunNumSet=10;
for RFCycleRun=1:RFRunNumSet
nTree=200;
nLeaf=5;
RFModel_forest=TreeBagger(nTree,N2O_X_train,N2O_Y_train,...
    'Method','regression','OOBPredictorImportance','on', 'MinLeafSize',nLeaf, ...
    'PredictorNames',{'MAT','MAP','BD','pH','SOC','TN','TP','SM','MBC','MBN','NO3','NH4','Nfer'});
[RFPredictYield,RFPredictConfidenceInterval]=predict(RFModel_forest,N2O_X_test);
% Accuracy of RF    
RFRMSE=sqrt(sum(sum((RFPredictYield-N2O_Y_test).^2))/size(N2O_Y_test,1));
if RFRMSE<1.16     %当RMSE满足<X条件时，模型将自动停止
    break;
end
end

%变量重要性记录
RFPredict_global = RFPredictYield;  %记录各生态系统数据，为global做准备
N2O_Y_test_global = N2O_Y_test;     %记录各生态系统数据，为global做准备
RI_forest = RFModel_forest.OOBPermutedPredictorDeltaError/max(RFModel_forest.OOBPermutedPredictorDeltaError);

%% 2.2 grassland
N2O_X_grassland = N2O_X(LandID == 4,:);
N2O_Y_grassland = N2O_Y(LandID == 4,:);

boot_id = (514:-1:1);  % 1y=223, 6m=242, 1m=483, total=514
boot_id = boot(boot_id,100);
Input = N2O_X_grassland(boot_id(:,i),:);
Output = N2O_Y_grassland(boot_id(:,i),:);

% 划分训练集和测试集
[ndata, D] = size(N2O_X_grassland);          %ndata样本数，D维数
R = randperm(ndata,round(0.2*ndata));      %1到n这些数随机打乱得到的一个随机数字序列作为索引
N2O_X_test = N2O_X_grassland(R,:);     %20%个数据点作为测试样本Xtest
N2O_Y_test = N2O_Y_grassland(R,:);
N2O_X_grassland(R,:) = [];
N2O_Y_grassland(R,:) = [];
N2O_X_train = N2O_X_grassland;           %剩下的数据作为训练样本Xtraining
N2O_Y_train = N2O_Y_grassland;

% 模型训练
RFRunNumSet=10;
for RFCycleRun=1:RFRunNumSet
nTree=50;
nLeaf=3;
RFModel_grassland=TreeBagger(nTree,N2O_X_train,N2O_Y_train,...
    'Method','regression','OOBPredictorImportance','on', 'MinLeafSize',nLeaf, ...
    'PredictorNames',{'MAT','MAP','BD','pH','SOC','TN','TP','SM','MBC','MBN','NO3','NH4','Nfer'});
[RFPredictYield,RFPredictConfidenceInterval]=predict(RFModel_grassland,N2O_X_test);
% Accuracy of RF    
RFRMSE=sqrt(sum(sum((RFPredictYield-N2O_Y_test).^2))/size(N2O_Y_test,1));
if RFRMSE<1.2    %当RMSE满足<X条件时，模型将自动停止
    break;
end
end

%变量记录
RFPredict_global = [RFPredict_global,
    RFPredictYield];  %记录各生态系统数据，为global做准备
N2O_Y_test_global = [N2O_Y_test_global,
    N2O_Y_test];     %记录各生态系统数据，为global做准备
RI_grassland = RFModel_grassland.OOBPermutedPredictorDeltaError/max(RFModel_grassland.OOBPermutedPredictorDeltaError);

%% 2.3 cropland (n=4356)
N2O_X_croplands = N2O_X(LandID == 1,:);
N2O_Y_croplands = N2O_Y(LandID == 1,:);

boot_id = (1:4356);  % 1y=2604, 6m=2914, 1m=4111, total=4356
boot_id = boot(boot_id,100);
Input = N2O_X_croplands(boot_id(:,i),:);
Output = N2O_Y_croplands(boot_id(:,i),:);

% 划分训练集和测试集
[ndata, D] = size(N2O_X_croplands);          %ndata样本数，D维数
R = randperm(ndata,round(0.2*ndata));   %1到n这些数随机打乱得到的一个随机数字序列作为索引
N2O_X_test = N2O_X_croplands(R,:);     %20%个数据点作为测试样本Xtest
N2O_Y_test = N2O_Y_croplands(R,:);
N2O_X_croplands(R,:) = [];
N2O_Y_croplands(R,:) = [];
N2O_X_train = N2O_X_croplands;           %剩下的数据作为训练样本Xtraining
N2O_Y_train = N2O_Y_croplands;

% 模型训练
RFRunNumSet=10;
for RFCycleRun=1:RFRunNumSet
nTree=100;
nLeaf=5;
RFModel_croplands=TreeBagger(nTree,N2O_X_train,N2O_Y_train,...
    'Method','regression','OOBPredictorImportance','on', 'MinLeafSize',nLeaf, ...
    'PredictorNames',{'MAT','MAP','BD','pH','SOC','TN','TP','SM','MBC','MBN','NO3','NH4','Nfer'});
[RFPredictYield,RFPredictConfidenceInterval]=predict(RFModel_croplands,N2O_X_test);

% Accuracy of RF    
RFRMSE=sqrt(sum(sum((RFPredictYield-N2O_Y_test).^2))/size(N2O_Y_test,1));
if RFRMSE<1.1      %当RMSE满足<X条件时，模型将自动停止
    break;
end
end

%变量记录
RFPredict_global = [RFPredict_global,
    RFPredictYield];  %记录各生态系统数据，为global做准备
N2O_Y_test_global = [N2O_Y_test_global,
    N2O_Y_test];     %记录各生态系统数据，为global做准备
RI_cropland = RFModel_croplands.OOBPermutedPredictorDeltaError/max(RFModel_croplands.OOBPermutedPredictorDeltaError);

% Accuracy of global performence
Global_RFR2 = 1-(sumsqr(N2O_Y_test_global-RFPredict_global)/sumsqr(N2O_Y_test_global-mean(N2O_Y_test_global)))

%% 3 模型预测
%% 3.1 土地利用数据导入和处理
% land cover data （2020年）
Input_way1 ='D:\研究生学习\氮循环\N N2O\data\MCD12C1.A2020001.061.2022172062638.hdf';    
		%hdf文件在电脑中的保存位置
Input_way2 ='/MOD12C1/Data Fields/Majority_Land_Cover_Type_1'; 
		%第一种分类数据在hdf文件中的位置         
Landcover2 = hdfread(Input_way1,Input_way2, 'Index', {[1  1],[1  1],[3600  7200]});
Landcover2 = imresize(Landcover2,[360,720],'nearest');%统一图像尺寸,采用最近邻插值算法
Landcover2 = reshape(Landcover2,259200,1);

% land cover data 2001
Input_way1 ='D:\研究生学习\氮循环\N N2O\data\MCD12C1.A2001001.061.2022146170409.hdf';    
		%hdf文件在电脑中的保存位置
Input_way2 ='/MOD12C1/Data Fields/Majority_Land_Cover_Type_1'; 
		%第一种分类数据在hdf文件中的位置         
Landcover_2001 = hdfread(Input_way1,Input_way2, 'Index', {[1  1],[1  1],[3600  7200]});
Landcover_2001 = imresize(Landcover_2001,[360,720],'nearest');%统一图像尺寸,采用最近邻插值算法
Landcover_2001 = reshape(Landcover_2001,259200,1);

% land cover data 2020
Input_way1 ='D:\研究生学习\氮循环\N N2O\data\MCD12C1.A2020001.061.2022172062638.hdf';    
		%hdf文件在电脑中的保存位置
Input_way2 ='/MOD12C1/Data Fields/Majority_Land_Cover_Type_1'; 
		%第一种分类数据在hdf文件中的位置         
Landcover_2020 = hdfread(Input_way1,Input_way2, 'Index', {[1  1],[1  1],[3600  7200]});
Landcover_2020 = imresize(Landcover_2020,[360,720],'nearest');%统一图像尺寸,采用最近邻插值算法
Landcover_2020 = reshape(Landcover_2020,259200,1);

%计算土地里利用转移矩阵
%设natural ecosystems为A，包括了forest和grassland，分类号有1~5，8~10
%设croplands为B，分类号有12~14

%找出两个时间段的分类信息
A_2001 = find(Landcover_2001 ==1 | Landcover_2001 == 2 | Landcover_2001 == 3 | Landcover_2001 == 4 ...
    | Landcover_2001 ==5 | Landcover_2001 ==8 | Landcover_2001 == 9 | Landcover_2001 == 10);
B_2001 = find(Landcover_2001 >= 12 & Landcover_2001 <= 14);
A_2020 = find(Landcover_2020 ==1 | Landcover_2020 == 2 | Landcover_2020 == 3 | Landcover_2020 == 4 ...
    | Landcover_2020 ==5 | Landcover_2020 ==8 | Landcover_2020 == 9 | Landcover_2020 == 10);
B_2020 = find(Landcover_2020 >= 12 & Landcover_2020 <= 14);
AtoB = intersect(A_2001,B_2020);    %取交集，找出由自然生态系统变为农田的区域,658
BtoA = intersect(B_2001,A_2020);    %取交集，找出由农田变为自然生态系统的区域,589

load X_predict.mat
X_predict = X_predict(:,[1:13]); %去掉一些预测变量
%% 3.2 模型预测
%% 3.2.1 2020年土地利用类型状态
% forest
N2O_predict_forest = single(nan*[1:259200]'); 
N2O_predict_forest(Landcover2 >=1 & Landcover2 <=17) = 0;
X1 = X_predict(Landcover2 == 1 | Landcover2 == 2 | Landcover2 == 3 | Landcover2 == 4 | Landcover2 == 5 ...
     | Landcover2 == 8 | Landcover2 == 9,:);
N2O_predict_forest(Landcover2 == 1 | Landcover2 == 2 | Landcover2 == 3 | Landcover2 == 4 | Landcover2 == 5 ...
     | Landcover2 == 8 | Landcover2 == 9,:) = predict(RFModel_forest,X1); 
N2O_predict_forest(N2O_predict_forest<0)=0;

% grassland N2O
N2O_predict_grassland = single(nan*[1:259200]'); 
N2O_predict_grassland(Landcover2 >=1 & Landcover2 <=17) = 0;
X1 = X_predict( Landcover2 == 10,:);
N2O_predict_grassland(Landcover2 == 10,:) = predict(RFModel_grassland,X1); 
N2O_predict_grassland(N2O_predict_grassland<0)=0;

% cropland N2O
N2O_predict_croplands = single(nan*[1:259200]');
N2O_predict_croplands(Landcover2 >=1 & Landcover2 <=17) = 0;
X1 = X_predict(Landcover2 >= 12 & Landcover2 <= 14,:);
N2O_predict_croplands(Landcover2 >= 12 & Landcover2 <= 14,:) = predict(RFModel_croplands,X1); 
N2O_predict_croplands(N2O_predict_croplands<0)=0;

% 合并生态类型
N2O_predict2 = single(nan*[1:259200]'); %创建空数据集
N2O_predict2(Landcover2 >=1 & Landcover2 <=17) = 0;
N2O_predict2(Landcover2 == 1 | Landcover2 == 2 | Landcover2 == 3 | Landcover2 == 4 | Landcover2 == 5 ...
     | Landcover2 == 8 | Landcover2 == 9) = N2O_predict_forest(Landcover2 == 1 | Landcover2 == 2 | Landcover2 == 3 | Landcover2 == 4 | Landcover2 == 5 ...
     | Landcover2 == 8 | Landcover2 == 9);
N2O_predict2(Landcover2 == 10) = N2O_predict_grassland(Landcover2 == 10);
N2O_predict2(Landcover2 >= 12 & Landcover2 <= 14) = N2O_predict_croplands(Landcover2 >= 12 & Landcover2 <= 14);

N2O_predict_natural = single(nan*[1:259200]');   %选取自然生态系统
N2O_predict_natural(Landcover2 == 1 | Landcover2 == 2 | Landcover2 == 3 | Landcover2 == 4 | Landcover2 == 5 ...
     | Landcover2 == 8 | Landcover2 == 9) = N2O_predict_forest(Landcover2 == 1 | Landcover2 == 2 | Landcover2 == 3 | Landcover2 == 4 | Landcover2 == 5 ...
     | Landcover2 == 8 | Landcover2 == 9);
N2O_predict_natural(Landcover2 == 10) = N2O_predict_grassland(Landcover2 == 10);

%% 2001年土地利用类型状态
% forest
N2O_2001_forest = single(nan*[1:259200]'); 
N2O_2001_forest(Landcover_2001 >=1 & Landcover_2001 <=17) = 0;
X1 = X_predict(Landcover_2001 == 1 | Landcover_2001 == 2 | Landcover_2001 == 3 | Landcover_2001 == 4 | Landcover_2001 == 5 ...
     | Landcover_2001 == 8 | Landcover_2001 == 9,:);
N2O_2001_forest(Landcover_2001 == 1 | Landcover_2001 == 2 | Landcover_2001 == 3 | Landcover_2001 == 4 | Landcover_2001 == 5 ...
     | Landcover_2001 == 8 | Landcover_2001 == 9,:) = predict(RFModel_forest,X1); 
N2O_2001_forest(N2O_2001_forest<0)=0;

% grassland N2O
N2O_2001_grassland = single(nan*[1:259200]'); 
N2O_2001_grassland(Landcover_2001 >=1 & Landcover_2001 <=17) = 0;
X1 = X_predict( Landcover_2001 == 10,:);
N2O_2001_grassland(Landcover_2001 == 10,:) = predict(RFModel_grassland,X1); 
N2O_2001_grassland(N2O_2001_grassland <0)=0;

% cropland N2O
N2O_2001_croplands = single(nan*[1:259200]');
N2O_2001_croplands(Landcover_2001 >=1 & Landcover_2001 <=17) = 0;
X1 = X_predict(Landcover_2001 >= 12 & Landcover_2001 <= 14,:);
N2O_2001_croplands(Landcover_2001 >= 12 & Landcover_2001 <= 14,:) = predict(RFModel_croplands,X1); 
N2O_2001_croplands(N2O_2001_croplands<0)=0;

% 合并生态类型
N2O_predict2001 = single(nan*[1:259200]'); %创建空数据集
N2O_predict2001(Landcover_2001 >=1 & Landcover_2001 <=17) = 0;
N2O_predict2001(Landcover_2001 == 1 | Landcover_2001 == 2 | Landcover_2001 == 3 | Landcover_2001 == 4 | Landcover_2001 == 5 ...
     | Landcover_2001 == 8 | Landcover_2001 == 9) = N2O_2001_forest(Landcover_2001 == 1 | Landcover_2001 == 2 | Landcover_2001 == 3 | Landcover_2001 == 4 | Landcover_2001 == 5 ...
     | Landcover_2001 == 8 | Landcover_2001 == 9);
N2O_predict2001(Landcover_2001 == 10) = N2O_2001_grassland(Landcover_2001 == 10);
N2O_predict2001(Landcover_2001 >= 12 & Landcover_2001 <= 14) = N2O_2001_croplands(Landcover_2001 >= 12 &Landcover_2001 <= 14);

N2O_2001_natural = single(nan*[1:259200]');   %选取自然生态系统
N2O_2001_natural(Landcover_2001 == 1 | Landcover_2001 == 2 | Landcover_2001 == 3 | Landcover_2001 == 4 | Landcover_2001 == 5 ...
     | Landcover_2001 == 8 | Landcover_2001 == 9) = N2O_2001_forest(Landcover_2001 == 1 | Landcover_2001 == 2 | Landcover_2001 == 3 | Landcover_2001 == 4 | Landcover_2001 == 5 ...
     | Landcover_2001 == 8 | Landcover_2001 == 9);
N2O_2001_natural(Landcover_2001 == 10) = N2O_2001_grassland(Landcover_2001 == 10);

%% 生态类型变化部分
Land_AtoB_2020 = single(nan*[1:259200]');
Land_BtoA_2020 = single(nan*[1:259200]');
Land_AtoB_2001 = single(nan*[1:259200]');
Land_BtoA_2001 = single(nan*[1:259200]');

% 算2020年的值
%情况一，2000natural变为2020cropland
Land_AtoB_2020(AtoB) = N2O_predict_croplands(AtoB);     %496个数据，相比658少了
%情况二，2000cropland变为2020natural
Land_BtoA_2020(BtoA) = N2O_predict_natural(BtoA);      % 498个数据

% 算2000年的值
%情况一，2022年是cropland 但 2000natural的值
Land_AtoB_2001(AtoB) = N2O_2001_natural(AtoB);   
%情况二，2022年是natural 但 2000cropland的值
Land_BtoA_2001(BtoA) = N2O_2001_croplands(BtoA);     

% 提取每次循环的信息
N2O_100(:,i) = N2O_predict2;   %从log（ug*m-2*day-1)换算成g*m-2*y-1
N2O_natural_100(:,i) = N2O_predict_natural;
N2O_forest_100(:,i) = N2O_predict_forest;
N2O_grassland_100(:,i) = N2O_predict_grassland;
N2O_cropland_100(:,i) = N2O_predict_croplands;

Land_AtoB_100_2020(:,i) = Land_AtoB_2020;
Land_BtoA_100_2020(:,i) = Land_BtoA_2020;
Land_AtoB_100_2001(:,i) = Land_AtoB_2001;
Land_BtoA_100_2001(:,i) = Land_BtoA_2001;

RI_forest_100(:,i) = RI_forest';
RI_grassland_100(:,i) = RI_grassland';
RI_cropland_100(:,i) = RI_cropland';

Global_RFR2_100(:,i) = Global_RFR2
% {'MAT','MAP','BD','pH','SOC','TN','TP','SM','MBC','MBN','NO3','NH4'}
end

RI_forest_mean = mean(RI_forest_100,2); %提取变量重要性信息
RI_grassland_mean = mean(RI_grassland_100,2);
RI_croplands_mean = mean(RI_cropland_100,2);

save meanvalue_0830 N2O_100 N2O_natural_100 N2O_forest_100 N2O_grassland_100 N2O_cropland_100...
    RI_forest_mean RI_grassland_mean RI_croplands_mean Global_RFR2_100 Landcover2

% load meanvalue.mat
%% 4 pattern和uncertainty图
N2O_forest_mean = mean(N2O_forest_100,2,"omitnan");
N2O_grassland_mean = mean(N2O_grassland_100,2,"omitnan");
N2O_cropland_mean = mean(N2O_cropland_100,2,"omitnan");

N2O_mean = mean(N2O_100,2,"omitnan");
N2O_std = std(N2O_100,1,2,"omitnan");

N2O_100t = exp(N2O_100)*365*0.000001;
N2O_mean000 = mean(N2O_100t,2,"omitnan");
N2O_std000 = std(N2O_100t,1,2,"omitnan");
N2O_CV = N2O_std000./N2O_mean000;
N2O_CV(Landcover2 == 6 | Landcover2 ==7| Landcover2 ==11| Landcover2 ==15| Landcover2 ==16| Landcover2 ==17) = 0;

N2O_forest_CV = single(nan*[1:259200]'); 
N2O_forest_CV(Landcover2 >=1 & Landcover2 <=17) = 0;
N2O_grassland_CV = single(nan*[1:259200]'); 
N2O_grassland_CV(Landcover2 >=1 & Landcover2 <=17) = 0;
N2O_cropland_CV = single(nan*[1:259200]'); 
N2O_cropland_CV(Landcover2 >=1 & Landcover2 <=17) = 0;
N2O_forest_CV(Landcover2 ==1 | Landcover2 == 2 | Landcover2 == 3 | Landcover2 == 4 | Landcover2 ==5 ...
    | Landcover2 ==8 | Landcover2 == 9) = N2O_CV(Landcover2 ==1 | Landcover2 == 2 | Landcover2 == 3 | Landcover2 == 4 | Landcover2 ==5 ...
    | Landcover2 ==8 | Landcover2 == 9);
N2O_grassland_CV(Landcover2 == 10) = N2O_CV(Landcover2 == 10);
N2O_cropland_CV(Landcover2 >= 12 & Landcover2 <= 14) = N2O_CV(Landcover2 >= 12 & Landcover2 <= 14);

%% 4.1 forest
% 基于Mapping toolbox画图
figure()
N2O_forest_meant = exp(N2O_forest_mean)*365*0.000001;%去对数,并转化为g m-2 yr-1
N2O_forest = reshape(N2O_forest_meant,[360,720]); 
lat = [-89.5:0.5:90];
lon = [-179.5:0.5:180];
[lon, lat] = meshgrid(lon,lat);
ax1 = axesm('MapProjection','pcarree','MapLatLimit',[-90 90],'MapLonLimit',[-180 180],'Frame','on','Grid','off', ...
    'FontName','Times','FontSize',12,'FEdgeColor','none', ...
    'MLineLocation',90,'MLabelRound', 0, 'MeridianLabel','on',...
    'PLineLocation',45,'PLabelRound', 0,'ParallelLabel','on','MLabelParallel','south');
tightmap;
% child1 = ax1.Children; 
% set(child1([3,4]), 'VerticalAlignment', 'baseline')
% 自设渐变色带
% mycolorpoint=[[112 7 38];...
%     [205 84 67];...
%     [247 188 158];...
%     [204 223 237];...
%     [61 143 193];...
%     [31 63 120];...
%     [230 230 230]];
mycolorpoint=[[207 223 39];...
    [110 212 74];...
    [48 176 127];...
    [34 144 145];...
    [43 112 145];...
    [63 70 140];...
    [63 37 84];...
    [230 230 230]];

% mycolorpoint=[[216 52 43];...
%     [252 141 88];...
%     [255 213 137];...
%     [242 249 218];...
%     [163 204 224];...
%     [83 128 187];...
%     [230 230 230]];

mycolorposition=[1 36 50 64 84 105 127 128];
mycolormap_r=interp1(mycolorposition,mycolorpoint(:,1),1:128,'linear','extrap');
mycolormap_g=interp1(mycolorposition,mycolorpoint(:,2),1:128,'linear','extrap');
mycolormap_b=interp1(mycolorposition,mycolorpoint(:,3),1:128,'linear','extrap');
mycolor=[mycolormap_r',mycolormap_g',mycolormap_b']/255;
mycolor=round(mycolor*10^4)/10^4;%保留4位小数

cm1 = colormap(ax1, mycolor);
cm1 = flipud(cm1);
colormap(ax1, cm1)
caxis([0 350*365*0.000001]);
N2O_forestf = flipud(N2O_forest);
i1 = surfm(lat, lon, N2O_forestf);
h1 = colorbar('FontName', 'Times', 'FontSize', 10);
print -djpeg -r600 N2O-map-forest

% forest CV
figure()
N2O_forest_uncertainty = reshape(N2O_forest_CV,[360,720]); 
ax5 = axesm('MapProjection','pcarree','MapLatLimit',[-90 90],'MapLonLimit',[-180 180],'Frame','on','Grid','off', ...
    'FontName','Times','FontSize',12,'FEdgeColor','none', ...
    'MLineLocation',90,'MLabelRound', 0, 'MeridianLabel','on',...
    'PLineLocation',45,'PLabelRound', 0,'ParallelLabel','on','MLabelParallel','south');
tightmap;
child1 = ax5.Children; 
set(child1([3,4]), 'VerticalAlignment', 'baseline')
cm1 = colormap(ax5, mycolor);
cm1 = flipud(cm1);
colormap(ax5, cm1)
caxis([0 0.45]);
N2O_forest_uncertaintyf = flipud(N2O_forest_uncertainty);
i1 = surfm(lat, lon, N2O_forest_uncertaintyf);
h1 = colorbar('FontName', 'Times', 'FontSize', 10);
print -djpeg -r600 N2O-uncertainty-forest


%% 4.2 grassland
figure()
N2O_grassland_meant = exp(N2O_grassland_mean)*365*0.000001;%去对数,并转化为g m-2 yr-1
N2O_grassland = reshape(N2O_grassland_meant,[360,720]); 
ax2 = axesm('MapProjection','pcarree','MapLatLimit',[-90 90],'MapLonLimit',[-180 180],'Frame','on','Grid','off', ...
    'FontName','Times','FontSize',12,'FEdgeColor','none', ...
    'MLineLocation',90,'MLabelRound', 0, 'MeridianLabel','on',...
    'PLineLocation',45,'PLabelRound', 0,'ParallelLabel','on','MLabelParallel','south');
tightmap;
child1 = ax2.Children; 
set(child1([3,4]), 'VerticalAlignment', 'baseline')
cm1 = colormap(ax2, mycolor);
cm1 = flipud(cm1);
colormap(ax2, cm1)
caxis([0 350*365*0.000001]);
N2O_grasslandf = flipud(N2O_grassland);
i1 = surfm(lat, lon, N2O_grasslandf);
h1 = colorbar('FontName', 'Times', 'FontSize', 10);
print -djpeg -r600 N2O-map-grassland

% grassland CV
figure()
N2O_grassland_uncertainty = reshape(N2O_grassland_CV,[360,720]); 
ax5 = axesm('MapProjection','pcarree','MapLatLimit',[-90 90],'MapLonLimit',[-180 180],'Frame','on','Grid','off', ...
    'FontName','Times','FontSize',12,'FEdgeColor','none', ...
    'MLineLocation',90,'MLabelRound', 0, 'MeridianLabel','on',...
    'PLineLocation',45,'PLabelRound', 0,'ParallelLabel','on','MLabelParallel','south');
tightmap;
child1 = ax5.Children; 
set(child1([3,4]), 'VerticalAlignment', 'baseline')
cm1 = colormap(ax5, mycolor);
cm1 = flipud(cm1);
colormap(ax5, cm1)
caxis([0 0.45]);
N2O_grassland_uncertaintyf = flipud(N2O_grassland_uncertainty);
i1 = surfm(lat, lon, N2O_grassland_uncertaintyf);
h1 = colorbar('FontName', 'Times', 'FontSize', 10);
print -djpeg -r600 N2O-uncertainty-grassland

%% 4.3 cropland
figure()
N2O_cropland_meant = exp(N2O_cropland_mean)*365*0.000001;%去对数,并转化为g m-2 yr-1
N2O_cropland = reshape(N2O_cropland_meant,[360,720]); 
ax3 = axesm('MapProjection','pcarree','MapLatLimit',[-90 90],'MapLonLimit',[-180 180],'Frame','on','Grid','off', ...
    'FontName','Times','FontSize',12,'FEdgeColor','none', ...
    'MLineLocation',90,'MLabelRound', 0, 'MeridianLabel','on',...
    'PLineLocation',45,'PLabelRound', 0,'ParallelLabel','on','MLabelParallel','south');
tightmap;
child1 = ax3.Children; 
set(child1([3,4]), 'VerticalAlignment', 'baseline')
cm1 = colormap(ax3, mycolor);
cm1 = flipud(cm1);
colormap(ax3, cm1)
caxis([0 350*365*0.000001]);
N2O_croplandf = flipud(N2O_cropland);
i1 = surfm(lat, lon, N2O_croplandf);
h1 = colorbar('FontName', 'Times', 'FontSize', 10);
print -djpeg -r600 N2O-map-cropland

% cropland CV
figure()
N2O_cropland_uncertainty = reshape(N2O_cropland_CV,[360,720]); 
ax5 = axesm('MapProjection','pcarree','MapLatLimit',[-90 90],'MapLonLimit',[-180 180],'Frame','on','Grid','off', ...
    'FontName','Times','FontSize',12,'FEdgeColor','none', ...
    'MLineLocation',90,'MLabelRound', 0, 'MeridianLabel','on',...
    'PLineLocation',45,'PLabelRound', 0,'ParallelLabel','on','MLabelParallel','south');
tightmap;
child1 = ax5.Children; 
set(child1([3,4]), 'VerticalAlignment', 'baseline')
cm1 = colormap(ax5, mycolor);
cm1 = flipud(cm1);
colormap(ax5, cm1)
caxis([0 0.45]);
N2O_cropland_uncertaintyf = flipud(N2O_cropland_uncertainty);
i1 = surfm(lat, lon, N2O_cropland_uncertaintyf);
h1 = colorbar('FontName', 'Times', 'FontSize', 10);
print -djpeg -r600 N2O-uncertainty-cropland

%% 4.4 gloabal
% gloabl N2O
figure()
N2O_meant = exp(N2O_mean)*365*0.000001; %去对数,并转化为g m-2 yr-1
N2O_global= reshape(N2O_meant,[360,720]); 
ax4 = axesm('MapProjection','pcarree','MapLatLimit',[-90 90],'MapLonLimit',[-180 180],'Frame','on','Grid','off', ...
    'FontName','Times','FontSize',12,'FEdgeColor','none', ...
    'MLineLocation',90,'MLabelRound', 0, 'MeridianLabel','on',...
    'PLineLocation',45,'PLabelRound', 0,'ParallelLabel','on','MLabelParallel','south');
tightmap;
child1 = ax4.Children; 
set(child1([3,4]), 'VerticalAlignment', 'baseline')

cm1 = colormap(ax4, mycolor);
cm1 = flipud(cm1);
colormap(ax4, cm1)
% caxis([4 6.3]);
caxis([0 350*365*0.000001]);
N2O_meanf = flipud(N2O_global);
i1 = surfm(lat, lon, N2O_meanf);
h1 = colorbar('FontName', 'Times', 'FontSize', 10);
% saveas(gcf,'D:\研究生学习\氮循环\N N2O\写作\STOEN返修补充分析\N2O-map-combined-100mean.png')
print -djpeg -r600 N2O-map-global

% global uncertainty CV
figure()
N2O_uncertainty = reshape(N2O_CV,[360,720]); 
ax5 = axesm('MapProjection','pcarree','MapLatLimit',[-90 90],'MapLonLimit',[-180 180],'Frame','on','Grid','off', ...
    'FontName','Times','FontSize',12,'FEdgeColor','none', ...
    'MLineLocation',90,'MLabelRound', 0, 'MeridianLabel','on',...
    'PLineLocation',45,'PLabelRound', 0,'ParallelLabel','on','MLabelParallel','south');
tightmap;
child1 = ax5.Children; 
set(child1([3,4]), 'VerticalAlignment', 'baseline')
cm1 = colormap(ax5, mycolor);
cm1 = flipud(cm1);
colormap(ax5, cm1)
%caxis([0 0.45]);
N2O_uncertaintyf = flipud(N2O_uncertainty);
i1 = surfm(lat, lon, N2O_uncertaintyf);
h1 = colorbar('FontName', 'Times', 'FontSize', 10);
print -djpeg -r600 N2O-uncertainty-global

%% 5.Land cover change N2O
Land_AtoB_2020_mean = mean(Land_AtoB_100_2020,2,"omitnan");
Land_BtoA_2020_mean = mean(Land_BtoA_100_2020,2,"omitnan");
Land_AtoB_2001_mean = mean(Land_AtoB_100_2001,2,"omitnan");
Land_BtoA_2001_mean = mean(Land_BtoA_100_2001,2,"omitnan");

Land_AtoB_2020 = reshape(Land_AtoB_2020_mean,[360,720]);
Land_BtoA_2020 = reshape(Land_BtoA_2020_mean,[360,720]);
Land_AtoB_2001 = reshape(Land_AtoB_2001_mean,[360,720]);
Land_BtoA_2001 = reshape(Land_BtoA_2001_mean,[360,720]);

% t_AtoB_2020 = Land_AtoB_2020.* Area_WGS_1984;
% t_BtoA_2020 = Land_BtoA_2020.* Area_WGS_1984;
% t_AtoB_2001 = Land_AtoB_2001.* Area_WGS_1984;
% t_BtoA_2001 = Land_BtoA_2001.* Area_WGS_1984;

% 统计分析：利用值和像元来算总量，进行对比变化前后速率以及变化量的对比
s_AtoB_2020 = Land_AtoB_2020(Land_AtoB_2020 >0);
s_BtoA_2020 = Land_BtoA_2020(Land_BtoA_2020 >0);
s_AtoB_2001 = Land_AtoB_2001(Land_AtoB_2001 >0);
s_BtoA_2001 = Land_BtoA_2001(Land_BtoA_2001 >0);

% 做变化密度图(2020年状态)
figure1 = figure
[lat_AtoB,lon_AtoB] =  find(Land_AtoB_2020>0);
lat_AtoBr = (lat_AtoB-180)/2;
lon_AtoBr = (lon_AtoB-360)/2;
[lat_BtoA,lon_BtoA] =  find(Land_BtoA_2020>0);
lat_BtoAr = (lat_BtoA-180)/2;
lon_BtoAr = (lon_BtoA-360)/2;

weights_AtoB = Land_AtoB_2020(Land_AtoB_2020 > 0);  %得到N2O排放信息
weights_BtoA = Land_BtoA_2020(Land_BtoA_2020 > 0); 
gx1 = geoaxes;
geolimits([-68.3872   78.0683],[-128.7875  156.2875])
geodensityplot(-lat_AtoBr,lon_AtoBr,weights_AtoB,'FaceColor',[181/255 24/255 8/255])
hold on
geodensityplot(-lat_BtoAr,lon_BtoAr,weights_BtoA,'FaceColor',[64/255 118/255 179/255])
gx1.Grid = 'off';gx1.Scalebar.Visible = 'off';geolimits('auto')
gx1.LatitudeLabel.String = 'Latitude'
gx1.LongitudeLabel.String = 'Longitude'
dp.FaceColor = 'interp';
title 'N2O emissions in land use change area';

print -djpeg -r600 N2Ochange
close all
% geobasemap grayland
% geobubble(-lat_AtoBr,lon_AtoBr,weights_AtoB)
