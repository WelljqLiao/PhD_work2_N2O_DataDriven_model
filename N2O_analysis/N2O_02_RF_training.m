% -*- coding: GBK -*-
% Created on June 27 2023 by Jiaqiang Liao
% 修改稿：区分观测时间，并加入施肥量，温度和降水的季节性补充
% 调试模型随机森林训练模块，为N2O_04_Final_anlysis做准备
%% 1.数据导入
clc,clear all

%% 1.1 N2O data read  
% (1)全6016数据
datapath = 'D:\研究生学习\氮循环\N N2O\data\';
%input = strcat(datapath,'Field nitrous oxide emission_by Lizhaolei.csv');
%input = strcat(datapath,'N2O_提取数据集.csv');
%原始数据+缺失值用数据产品插补↓
input = strcat(datapath,'Field nitrous oxide emission_by Lizhaolei - 补插值.csv');

N2O_u = xlsread(input);  %读取N2O原始数据库
N2O_u = fillmissing(N2O_u,'movmean',6016);%取均值替代缺失值
%N2O_u = fillmissing(N2O_u,"constant",nan); 
MAT = N2O_u(:,8); 
MAP = N2O_u(:,9);
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
Nf = N2O_u(:,33);
Temp_season = N2O_u(:,35);
Prep_season = N2O_u(:,36);

N2O_X = [MAT,MAP,BD,pH,SOC,TN,TP,SM,MBC,MBN,NO3,NH4,Nf,Temp_season,Prep_season];
N2O_y = N2O_u(:,28); 

% 取对数
N2O_Y = log(N2O_y);

% 去缺失值
% N2O_X = fillmissing(N2O_x,"constant",nan);
% N2O_Y = fillmissing(N2O_y,'constant',nan);
N2O = [N2O_X,N2O_Y];
Landcover = N2O_u(:,31); %读取土地利用类型(按坐标匹配modis)
LandID = N2O_u(:,32);   %读取土地利用类型(按原始搜集文献的数据区分)
Duration = N2O_u(:,34); %观测时间
weights = N2O_u(:,37); %时间取样权重（小于1月=0.1；1~6月=0.3，6~12月=0.6，12月以上=1）

%(2)大于1年的数据
N2O_1y = N2O(find(Duration >= 12),:);
Landcover_1y = Landcover(find(Duration >= 12),:);
LandID_1y = LandID(find(Duration >= 12),:);
%(3)大于6个月的数据
N2O_6m = N2O(find(Duration >= 6),:);
Landcover_6m = Landcover(find(Duration >= 6),:);
LandID_6m = LandID(find(Duration >= 6),:);
%(4)大于1个月的数据
N2O_1m = N2O(find(Duration >= 1),:);
Landcover_1m = Landcover(find(Duration >= 1),:);
LandID_1m = LandID(find(Duration >= 1),:);

save('N2O_database','N2O', 'Landcover', 'LandID',...
    'N2O_1y','N2O_6m','N2O_1m', ...
    'LandID_1y','LandID_6m','LandID_1m', ...
    'Landcover_1y','Landcover_6m','Landcover_1m','weights'); %保存输入数据

%% 1.2分训练集和测试集
[n, D] = size(N2O);
ID = 1:n;
samples = randsample(ID, n, true, weights); %权重取样，有放回
N2O = N2O(ID,:);
N2O_X = N2O(:,1:13);
N2O_Y = N2O(:,16);

[ndata, D] = size(N2O_X);                    %ndata样本数，D维数
R = randperm(ndata,round(0.2*ndata));        %1到n这些数随机打乱得到的一个随机数字序列作为索引
N2O_X_test = N2O_X(R,:);                     %以索引的前20%个数据点作为测试样本Xtest
N2O_Y_test = N2O_Y(R,:);
N2O_X(R,:) = [];
N2O_Y(R,:) = [];
N2O_X_train = N2O_X;           %剩下的数据作为训练样本Xtraining
N2O_Y_train = N2O_Y;

%% 2.构建全球总体随机森林

% 2.1 Number of Leaves and Trees Optimization

Input = N2O_X;
Output = N2O_Y;

% for RFOptimizationNum=1:5 % RFOptimizationNum是为了多次循环，防止最优结果受到随机干扰；
    
RFLeaf=[5,10,20,50];
% RFLeaf定义初始的叶子节点个数，我这里设置了从5到500，也就是从5到500这个范围内找到最优叶子节点个数。

col='rgbcmyk';
figure('Name','RF Leaves and Trees');
for i=1:length(RFLeaf)
    RFModel=TreeBagger(2000,Input,Output,'Method','regression','OOBPrediction','on','MinLeafSize',RFLeaf(i));
    plot(oobError(RFModel),col(i));
    hold on
end
xlabel('Number of Grown Trees');
ylabel('Mean Squared Error') ;
LeafTreelgd=legend({'5' '10' '20' '50' '100' '200' '500'},'Location','NorthEast');
title(LeafTreelgd,'Number of Leaves');
hold off;

% disp(RFOptimizationNum);
% end


% 2.2 Cycle Preparation &T raining Set and Test Set Division

RFScheduleBar=waitbar(0,'Random Forest is Solving...');
RFRMSEMatrix=[];
RFrAllMatrix=[];
RFRunNumSet=10;
% RFRMSEMatrix与RFrAllMatrix分别用来存放每一次运行的RMSE、r结果，
% RFRunNumSet是循环次数，也就是RF运行的次数。

for RFCycleRun=1:RFRunNumSet

nTree=100;
nLeaf=5;
RFModel=TreeBagger(nTree,N2O_X_train,N2O_Y_train, ...
    'Method','regression','OOBPredictorImportance','on', 'MinLeafSize',nLeaf, ...
    'PredictorNames',{'MAT','MAP','BD','pH','SOC','TN','TP','SM','MBC','MBN','NO3','NH4','Nfer'});
[RFPredictYield,RFPredictConfidenceInterval]=predict(RFModel,N2O_X_test);
% RFPredictYield是预测结果，RFPredictConfidenceInterval是预测结果的置信区间

figure()
plotregression(N2O_Y_test,RFPredictYield)
xlabel('Observerd N2O'),ylabel('Predict N2O')
legend('Random Forest','test data')
hold off

% Accuracy of RF    

RFRMSE=sqrt(sum(sum((RFPredictYield-N2O_Y_test).^2))/size(N2O_Y_test,1));
RFrMatrix=corrcoef(RFPredictYield,N2O_Y_test);
RFR2 = 1-(sumsqr(N2O_Y_test-RFPredictYield)/sumsqr(N2O_Y_test-mean(N2O_Y_test)));
RFr=RFrMatrix(1,2);
RFRMSEMatrix=[RFRMSEMatrix,RFRMSE];
RFrAllMatrix=[RFrAllMatrix,RFr];
if RFRMSE<1     %当RMSE满足<1.15这个条件时，模型将自动停止
    disp(RFRMSE);
    break;
end
disp(RFCycleRun);
str=['Random Forest is Solving...',num2str(100*RFCycleRun/RFRunNumSet),'%'];
waitbar(RFCycleRun/RFRunNumSet,RFScheduleBar,str);
end
close(RFScheduleBar);

% 2.3 Variable Importance Contrast 变量重要性排序

figure('Name','Variable Importance Contrast');
Relative_importance = RFModel.OOBPermutedPredictorDeltaError/max(RFModel.OOBPermutedPredictorDeltaError);
% RFModel.OOBPermutedPredictorDeltaError变量重要性程度
bar([1:13],Relative_importance)
xtickangle(45);
set(gca,'xticklabels',{'MAT','MAP','BD','pH','SOC','TN','TP','SM','MBC','MBN','NO3','NH4','Nfer'},'XDir','normal')
xlabel('Factor');
ylabel('Importance');

% Shapley方法
% RFModel = fitrtree(N2O_X_train,N2O_Y_train, 'MinParentSize',10);
% queryPoint = N2O_X_train(2,:);
% explainer1 = shapley(RFModel,N2O_X_train,'QueryPoint',queryPoint); 单点查询
% plot(explainer1)

RFModel_global = fitrtree(N2O_X_train,N2O_Y_train,'MinLeafSize',5,...
    'PredictorSelection','curvature','Surrogate','on',...
    'PredictorNames',{'MAT','MAP','BD','pH','SOC','TN','TP','SM','MBC','MBN','NO3','NH4','Nfer'}, ...
    'ResponseName','N2O emission');
[RFPredictYield,RFPredictConfidenceInterval]=predict(RFModel_global,N2O_X_test);

%模型效果拟合图
figure()
plotregression(N2O_Y_test,RFPredictYield)
xlabel('Observerd N2O'),ylabel('Predict N2O')
legend('Random Forest','test data','global')
hold off
print(gcf,'Regression_global0000','-dpng','-r600')

RFPredict_global = RFPredictYield;  %记录各生态系统数据，为global做准备
N2O_Y_test_global = N2O_Y_test;     %记录各生态系统数据，为global做准备

% Shapley value （耗时非常长，约2.5h）
for i = 1:4813
queryPoint = N2O_X_train(i,:);
explainer1 = shapley(RFModel_global,N2O_X_train,'QueryPoint',queryPoint,'UseParallel',true);
shapleyvalue(:,i) = explainer1.ShapleyValues(:,"ShapleyValue");
end

save globalshapley shapleyvalue  %保存全局shapley值

load globalshapley.mat
shapley_p0 = table2array(shapleyvalue);
shapley_p = abs(shapley_p0); %绝对值
shapley_mean = mean(shapley_p,1);

%用shapley vaule和特征的协方差来判断贡献的正负
shapley_p0 = shapley_p0';
MAT_cov = cov(shapley_p0(:,1),N2O_X_train(:,1))
MAP_cov = cov(shapley_p0(:,2),N2O_X_train(:,2))
BD_cov = cov(shapley_p0(:,3),N2O_X_train(:,3))
pH_cov = cov(shapley_p0(:,4),N2O_X_train(:,4))
SOC_cov = cov(shapley_p0(:,5),N2O_X_train(:,5))
TN_cov = cov(shapley_p0(:,6),N2O_X_train(:,6))
TP_cov = cov(shapley_p0(:,7),N2O_X_train(:,7))
SM_cov = cov(shapley_p0(:,8),N2O_X_train(:,8))
MBC_cov = cov(shapley_p0(:,9),N2O_X_train(:,9))
MBN_cov = cov(shapley_p0(:,10),N2O_X_train(:,10))
NO3_cov = cov(shapley_p0(:,11),N2O_X_train(:,11))
NH4_cov = cov(shapley_p0(:,12),N2O_X_train(:,12))
Nfer_cov = cov(shapley_p0(:,13),N2O_X_train(:,13))

figure()
bar(shapley_mean);
set(gca,'xticklabels',{'MAT','MAP','BD','pH','SOC','TN','TP','SM','MBC','MBN','NO3','NH4','Nfer'},'XDir','normal')
% plot(explainer1,'NumImportantPredictors',12)
print(gcf,'Shapley_global','-dpng','-r600')

%% 2.4 RF Model Storage
RFModelSavePath='D:\研究生学习\氮循环\N N2O\写作\STOEN返修补充分析\';
save(sprintf('%sRFmodel_N2O.mat',RFModelSavePath),'nLeaf','nTree',...
    'RFModel','RFPredictConfidenceInterval','RFPredictYield','RFr','RFR2','RFRMSE','Relative_importance',...
    'N2O_X_test','N2O_Y_test','N2O_X_train','N2O_Y_train');

subplot(3,5,1) 
plotPartialDependence(RFModel,1)
xlabel('MAT'),ylabel('Partial dependence'),title(' ')
subplot(3,5,2) 
plotPartialDependence(RFModel,2)
xlabel('MAP'),ylabel('Partial dependence'),title(' ')
subplot(3,5,3) 
plotPartialDependence(RFModel,3)
xlabel('BD'),ylabel('Partial dependence'),title(' ')
subplot(3,5,4) 
plotPartialDependence(RFModel,4)
xlabel('pH'),ylabel('Partial dependence'),title(' ')
subplot(3,5,5) 
plotPartialDependence(RFModel,5)
xlabel('SOC'),ylabel('Partial dependence'),title(' ')
subplot(3,5,6) 
plotPartialDependence(RFModel,6)
xlabel('TN'),ylabel('Partial dependence'),title(' ')
subplot(3,5,7) 
plotPartialDependence(RFModel,7)
xlabel('TP'),ylabel('Partial dependence'),title(' ')
subplot(3,5,8) 
plotPartialDependence(RFModel,8)
xlabel('SM'),ylabel('Partial dependence'),title(' ')
subplot(3,5,9) 
plotPartialDependence(RFModel,9)
xlabel('MBC'),ylabel('Partial dependence'),title(' ')
subplot(3,5,10) 
plotPartialDependence(RFModel,10)
xlabel('MBN'),ylabel('Partial dependence'),title(' ')
subplot(3,5,11) 
plotPartialDependence(RFModel,11)
xlabel('NO3'),ylabel('Partial dependence'),title(' ')
subplot(3,5,12) 
plotPartialDependence(RFModel,12)
xlabel('NH4'),ylabel('Partial dependence'),title(' ')
subplot(3,5,13) 
plotPartialDependence(RFModel,13)
xlabel('Nfer'),ylabel('Partial dependence'),title(' ')

print(gcf,'Partial dependce_global','-dpng','-r600')
% 2.5 partial dependence
numPoints = 10;
ptX = linspace(min(N2O_X_train(:,1)),max(N2O_X_train(:,1)),numPoints)';
ptY = linspace(min(N2O_X_train(:,2)),max(N2O_X_train(:,2)),numPoints)';
[pd,x,y] = partialDependence(RFModel,{'MAT','MAP'},'QueryPoints',[ptX ptY]);

t = tiledlayout(5,5,'TileSpacing','compact');

ax1 = nexttile(2,[4,4]);
imagesc(x,y,pd),title('Partial Dependence Plot'),colorbar('eastoutside'),ax1.YDir = 'normal';

ax2 = nexttile(22,[1,4]);
dX = diff(ptX(1:2));
edgeX = [ptX-dX/2;ptX(end)+dX];histogram(N2O_X_train(:,1),edgeX);xlabel('MAT'),xlim(ax1.XLim);

ax3 = nexttile(1,[4,1]);
dY = diff(ptY(1:2));
edgeY = [ptY-dY/2;ptY(end)+dY];
histogram(N2O_X_train(:,2),edgeY)
xlabel('MAP'),xlim(ax1.YLim);ax3.XDir = 'reverse';camroll(-90)

%% 3.分生态类型进行模型训练

% 基于原始N2O数据集生态区分类(croplands,desert,forest,grassland,wetland)训练；
% 匹配MODIS LANDCOVER PRODUCT进行对应的生态类型进行预测，对应划分如下：
% forest 1~5
% grassland 8~10
% croplands 12~14
% 其余土地利用类型由于没有对应数据，赋值为nan

clear all
load N2O_database.mat % 载入训练数据集(6016)
N2O_X = N2O(:,1:13);
N2O_Y = N2O(:,14);

%%  3.1 forest (n=679)
N2O_X_forest = N2O_X(LandID == 3,:);
N2O_Y_forest = N2O_Y(LandID == 3,:);
Input = N2O_X_forest;
Output = N2O_Y_forest;

% 划分训练集和测试集
rng(1998)
[ndata, D] = size(N2O_X_forest);          %ndata样本数，D维数
R = randperm(ndata,round(0.2*ndata));      %1到n这些数随机打乱得到的一个随机数字序列作为索引
N2O_X_test = N2O_X_forest(R,:);     %20%个数据点作为测试样本Xtest
N2O_Y_test = N2O_Y_forest(R,:);
N2O_X_forest(R,:) = [];
N2O_Y_forest(R,:) = [];
N2O_X_train = N2O_X_forest;           %剩下的数据作为训练样本Xtraining
N2O_Y_train = N2O_Y_forest;

%（1）TreeBagger的方法
RFScheduleBar=waitbar(0,'Random Forest is Solving...');
RFRMSEMatrix=[];
RFrAllMatrix=[];
RFRunNumSet=10;

for RFCycleRun=1:RFRunNumSet
nTree=100;
nLeaf=5;
RFModel_forest=TreeBagger(nTree,N2O_X_train,N2O_Y_train,...
    'Method','regression','OOBPredictorImportance','on', 'MinLeafSize',nLeaf,...
    'PredictorNames',{'MAT','MAP','BD','pH','SOC','TN','TP','SM','MBC','MBN','NO3','NH4','Nfer'});
[RFPredictYield,RFPredictConfidenceInterval]=predict(RFModel_forest,N2O_X_test);

RFPredict_global = RFPredictYield;
N2O_Y_test_global = N2O_Y_test;

figure()
plotregression(N2O_Y_test,RFPredictYield)
xlabel('Observerd N2O'),ylabel('Predict N2O')
legend('Random Forest','test data','forest')
hold off
print(gcf,'Regression_forest','-dpng','-r600')

% Accuracy of RF    
RFRMSE=sqrt(sum(sum((RFPredictYield-N2O_Y_test).^2))/size(N2O_Y_test,1));
RFrMatrix=corrcoef(RFPredictYield,N2O_Y_test);
RFR2 = 1-(sumsqr(N2O_Y_test-RFPredictYield)/sumsqr(N2O_Y_test-mean(N2O_Y_test)));
RFr=RFrMatrix(1,2);
RFRMSEMatrix=[RFRMSEMatrix,RFRMSE];
RFrAllMatrix=[RFrAllMatrix,RFr];
if RFRMSE<1.2    %当RMSE满足<X条件时，模型将自动停止
    disp(RFRMSE);
    break;
end
disp(RFCycleRun);
str=['Random Forest is Solving...',num2str(100*RFCycleRun/RFRunNumSet),'%'];
waitbar(RFCycleRun/RFRunNumSet,RFScheduleBar,str);
end
close(RFScheduleBar);

% 变量重要性排序
figure('Name','Variable Importance Contrast');
RI_forest = RFModel_forest.OOBPermutedPredictorDeltaError/max(RFModel_forest.OOBPermutedPredictorDeltaError);
bar([1:13],RI_forest)
xtickangle(45);
set(gca,'xticklabels',{'MAT','MAP','BD','pH','SOC','TN','TP','SM','MBC','MBN','NO3','NH4','Nfer'},'XDir','normal')
xlabel('Factor');
ylabel('Importance');


%% 3.1模型保存和偏向关分析
RFModelSavePath='D:\研究生学习\氮循环\N N2O\写作\STOEN返修补充分析\';
save(sprintf('%sRFmodel_N2O_forest.mat',RFModelSavePath),'nLeaf','nTree',...
    'RFModel_forest','RFPredictConfidenceInterval','RFPredictYield','RFr','RFR2','RFRMSE','RI_forest',...
    'N2O_X_test','N2O_Y_test','N2O_X_train','N2O_Y_train');

% partial dependence(2维)
numPoints = 10;
ptX = linspace(min(N2O_X_train(:,1)),max(N2O_X_train(:,1)),numPoints)';
ptY = linspace(min(N2O_X_train(:,2)),max(N2O_X_train(:,2)),numPoints)';
[pd,x,y] = partialDependence(RFModel_forest,{'MAT','MAP'},'QueryPoints',[ptX ptY]);

t = tiledlayout(5,5,'TileSpacing','compact');
ax1 = nexttile(1,[5,5]);
imagesc(x,y,pd),title('Partial Dependence Plot'),colorbar('eastoutside'),ax1.YDir = 'normal';
xlabel('MAT'),ylabel('MAP')
print(gcf,'Partial dependce_forest_MATMAP','-dpng','-r600')
%
numPoints = 10;
ptX = linspace(min(N2O_X_train(:,1)),max(N2O_X_train(:,1)),numPoints)';
ptY = linspace(min(N2O_X_train(:,11)),max(N2O_X_train(:,11)),numPoints)';
[pd,x,y] = partialDependence(RFModel_forest,{'MAT','NO3'},'QueryPoints',[ptX ptY]);

t = tiledlayout(5,5,'TileSpacing','compact');
ax1 = nexttile(1,[5,5]);
imagesc(x,y,pd),title('Partial Dependence Plot'),colorbar('eastoutside'),ax1.YDir = 'normal';
xlabel('MAT'),ylabel('NO3')
print(gcf,'Partial dependce_forest_MATNO3','-dpng','-r600')

%
numPoints = 10;
ptX = linspace(min(N2O_X_train(:,2)),max(N2O_X_train(:,2)),numPoints)';
ptY = linspace(min(N2O_X_train(:,11)),max(N2O_X_train(:,11)),numPoints)';
[pd,x,y] = partialDependence(RFModel_forest,{'MAT','NO3'},'QueryPoints',[ptX ptY]);

t = tiledlayout(5,5,'TileSpacing','compact');
ax1 = nexttile(1,[5,5]);
imagesc(x,y,pd),title('Partial Dependence Plot'),colorbar('eastoutside'),ax1.YDir = 'normal';
xlabel('MAP'),ylabel('NO3')
print(gcf,'Partial dependce_forest_MAPNO3','-dpng','-r600')

%% 3.2 grassland(n=514)
N2O_X_grassland = N2O_X(LandID == 4,:);
N2O_Y_grassland = N2O_Y(LandID == 4,:);
Input = N2O_X_grassland;
Output = N2O_Y_grassland;

% 划分训练集和测试集
rng(1998)
[ndata, D] = size(N2O_X_grassland);          %ndata样本数，D维数
R = randperm(ndata,round(0.2*ndata));      %1到n这些数随机打乱得到的一个随机数字序列作为索引
N2O_X_test = N2O_X_grassland(R,:);     %20%个数据点作为测试样本Xtest
N2O_Y_test = N2O_Y_grassland(R,:);
N2O_X_grassland(R,:) = [];
N2O_Y_grassland(R,:) = [];
N2O_X_train = N2O_X_grassland;           %剩下的数据作为训练样本Xtraining
N2O_Y_train = N2O_Y_grassland;

% （1）TreeBagger的方法
RFScheduleBar=waitbar(0,'Random Forest is Solving...');
RFRMSEMatrix=[];
RFrAllMatrix=[];
RFRunNumSet=10;

for RFCycleRun=1:RFRunNumSet
nTree=100;
nLeaf=5;
RFModel_grassland=TreeBagger(nTree,N2O_X_train,N2O_Y_train,...
    'Method','regression','OOBPredictorImportance','on', 'MinLeafSize',nLeaf,...
    'PredictorNames',{'MAT','MAP','BD','pH','SOC','TN','TP','SM','MBC','MBN','NO3','NH4','Nfer'});
[RFPredictYield,RFPredictConfidenceInterval]=predict(RFModel_grassland,N2O_X_test);

RFPredict_global = [RFPredict_global
    RFPredictYield];
N2O_Y_test_global = [N2O_Y_test_global
    N2O_Y_test];
figure()
plotregression(N2O_Y_test,RFPredictYield)
xlabel('Observerd N2O'),ylabel('Predict N2O')
legend('Random Forest','test data','grassland')
hold off
print(gcf,'Regression_grassland','-dpng','-r600')

% Accuracy of RF    
RFRMSE=sqrt(sum(sum((RFPredictYield-N2O_Y_test).^2))/size(N2O_Y_test,1));
RFrMatrix=corrcoef(RFPredictYield,N2O_Y_test);
RFR2 = 1-(sumsqr(N2O_Y_test-RFPredictYield)/sumsqr(N2O_Y_test-mean(N2O_Y_test)));
RFr=RFrMatrix(1,2);
RFRMSEMatrix=[RFRMSEMatrix,RFRMSE];
RFrAllMatrix=[RFrAllMatrix,RFr];
if RFRMSE<1.2    %当RMSE满足<X条件时，模型将自动停止
    disp(RFRMSE);
    break;
end
disp(RFCycleRun);
str=['Random Forest is Solving...',num2str(100*RFCycleRun/RFRunNumSet),'%'];
waitbar(RFCycleRun/RFRunNumSet,RFScheduleBar,str);
end
close(RFScheduleBar);

% 变量重要性排序
figure('Name','Variable Importance Contrast');
RI_grassland = RFModel_grassland.OOBPermutedPredictorDeltaError/max(RFModel_grassland.OOBPermutedPredictorDeltaError);
bar([1:13],RI_grassland)
xtickangle(45);
set(gca,'xticklabels',{'MAT','MAP','BD','pH','SOC','TN','TP','SM','MBC','MBN','NO3','NH4','Nfer'},'XDir','normal')
xlabel('Factor');
ylabel('Importance');

%% 3.2模型保存和偏向关分析
RFModelSavePath='D:\研究生学习\氮循环\N N2O\写作\STOEN返修补充分析\';
save(sprintf('%sRFmodel_N2O_grassland.mat',RFModelSavePath),'nLeaf','nTree',...
    'RFModel_grassland','RFPredictConfidenceInterval','RFPredictYield','RFr','RFR2','RFRMSE','RI_grassland',...
    'N2O_X_test','N2O_Y_test','N2O_X_train','N2O_Y_train');

% partial dependence
numPoints = 10;
ptX = linspace(min(N2O_X_train(:,1)),max(N2O_X_train(:,1)),numPoints)';
ptY = linspace(min(N2O_X_train(:,2)),max(N2O_X_train(:,2)),numPoints)';
[pd,x,y] = partialDependence(RFModel_grassland,{'MAT','MAP'},'QueryPoints',[ptX ptY]);

t = tiledlayout(5,5,'TileSpacing','compact');
ax1 = nexttile(1,[5,5]);
imagesc(x,y,pd),title('Partial Dependence Plot'),colorbar('eastoutside'),ax1.YDir = 'normal';
xlabel('MAT'),ylabel('MAP')
print(gcf,'Partial dependce_grassland_MATMAP','-dpng','-r600')
%
numPoints = 10;
ptX = linspace(min(N2O_X_train(:,1)),max(N2O_X_train(:,1)),numPoints)';
ptY = linspace(min(N2O_X_train(:,11)),max(N2O_X_train(:,11)),numPoints)';
[pd,x,y] = partialDependence(RFModel_grassland,{'MAT','NO3'},'QueryPoints',[ptX ptY]);

t = tiledlayout(5,5,'TileSpacing','compact');
ax1 = nexttile(1,[5,5]);
imagesc(x,y,pd),title('Partial Dependence Plot'),colorbar('eastoutside'),ax1.YDir = 'normal';
xlabel('MAT'),ylabel('NO3')
print(gcf,'Partial dependce_grassland_MATNO3','-dpng','-r600')

%
numPoints = 10;
ptX = linspace(min(N2O_X_train(:,2)),max(N2O_X_train(:,2)),numPoints)';
ptY = linspace(min(N2O_X_train(:,11)),max(N2O_X_train(:,11)),numPoints)';
[pd,x,y] = partialDependence(RFModel_grassland,{'MAT','NO3'},'QueryPoints',[ptX ptY]);

t = tiledlayout(5,5,'TileSpacing','compact');
ax1 = nexttile(1,[5,5]);
imagesc(x,y,pd),title('Partial Dependence Plot'),colorbar('eastoutside'),ax1.YDir = 'normal';
xlabel('MAP'),ylabel('NO3')
print(gcf,'Partial dependce_grassland_MAPNO3','-dpng','-r600')

%% 3.3 croplands/management (n=4356)
N2O_X_croplands = N2O_X(LandID == 1,:);
N2O_Y_croplands = N2O_Y(LandID == 1,:);
Input = N2O_X_croplands;
Output = N2O_Y_croplands;

% 划分训练集和测试集
rng(1998)
[ndata, D] = size(N2O_X_croplands);          %ndata样本数，D维数
R = randperm(ndata,round(0.2*ndata));      %1到n这些数随机打乱得到的一个随机数字序列作为索引
N2O_X_test = N2O_X_croplands(R,:);     %20%个数据点作为测试样本Xtest
N2O_Y_test = N2O_Y_croplands(R,:);
N2O_X_croplands(R,:) = [];
N2O_Y_croplands(R,:) = [];
N2O_X_train = N2O_X_croplands;           %剩下的数据作为训练样本Xtraining
N2O_Y_train = N2O_Y_croplands;

%（1）TreeBagger的方法
RFScheduleBar=waitbar(0,'Random Forest is Solving...');
RFRMSEMatrix=[];
RFrAllMatrix=[];
RFRunNumSet=10;

for RFCycleRun=1:RFRunNumSet
nTree=100;
nLeaf=5;
RFModel_croplands=TreeBagger(nTree,N2O_X_train,N2O_Y_train,...
    'Method','regression','OOBPredictorImportance','on', 'MinLeafSize',nLeaf,...
    'PredictorNames',{'MAT','MAP','BD','pH','SOC','TN','TP','SM','MBC','MBN','NO3','NH4','Nfer'});
[RFPredictYield,RFPredictConfidenceInterval]=predict(RFModel_croplands,N2O_X_test);

RFPredict_global = [RFPredict_global
    RFPredictYield];
N2O_Y_test_global = [N2O_Y_test_global
    N2O_Y_test];

figure()
plotregression(N2O_Y_test,RFPredictYield)
xlabel('Observerd N2O'),ylabel('Predict N2O')
legend('Random Forest','test data','croplands')
hold off
print(gcf,'Regression_cropland','-dpng','-r600')

% Accuracy of RF    
RFRMSE=sqrt(sum(sum((RFPredictYield-N2O_Y_test).^2))/size(N2O_Y_test,1));
RFrMatrix=corrcoef(RFPredictYield,N2O_Y_test);
RFR2 = 1-(sumsqr(N2O_Y_test-RFPredictYield)/sumsqr(N2O_Y_test-mean(N2O_Y_test)));
RFr=RFrMatrix(1,2);
RFRMSEMatrix=[RFRMSEMatrix,RFRMSE];
RFrAllMatrix=[RFrAllMatrix,RFr];
if RFRMSE<1.2    %当RMSE满足<X条件时，模型将自动停止
    disp(RFRMSE);
    break;
end
disp(RFCycleRun);
str=['Random Forest is Solving...',num2str(100*RFCycleRun/RFRunNumSet),'%'];
waitbar(RFCycleRun/RFRunNumSet,RFScheduleBar,str);
end
close(RFScheduleBar);

% 变量重要性排序
figure('Name','Variable Importance Contrast');
RI_croplands = RFModel_croplands.OOBPermutedPredictorDeltaError/max(RFModel_croplands.OOBPermutedPredictorDeltaError);
bar([1:13],RI_croplands)
xtickangle(45);
set(gca,'xticklabels',{'MAT','MAP','BD','pH','SOC','TN','TP','SM','MBC','MBN','NO3','NH4','Nfer'},'XDir','normal')
xlabel('Factor');
ylabel('Importance');

%% 3.3模型保存和偏向关分析
RFModelSavePath='D:\研究生学习\氮循环\N N2O\写作\STOEN返修补充分析\';
save(sprintf('%sRFmodel_N2O_croplands.mat',RFModelSavePath),'nLeaf','nTree',...
    'RFModel_croplands','RFPredictConfidenceInterval','RFPredictYield','RFr','RFR2','RFRMSE','RI_croplands',...
    'N2O_X_test','N2O_Y_test','N2O_X_train','N2O_Y_train');

% partial dependence
numPoints = 10;
ptX = linspace(min(N2O_X_train(:,1)),max(N2O_X_train(:,1)),numPoints)';
ptY = linspace(min(N2O_X_train(:,2)),max(N2O_X_train(:,2)),numPoints)';
[pd,x,y] = partialDependence(RFModel_croplands,{'MAT','MAP'},'QueryPoints',[ptX ptY]);

t = tiledlayout(5,5,'TileSpacing','compact');
ax1 = nexttile(1,[5,5]);
imagesc(x,y,pd),title('Partial Dependence Plot'),colorbar('eastoutside'),ax1.YDir = 'normal';
xlabel('MAT'),ylabel('MAP')
print(gcf,'Partial dependce_croplands_MATMAP','-dpng','-r600')
%
numPoints = 10;
ptX = linspace(min(N2O_X_train(:,1)),max(N2O_X_train(:,1)),numPoints)';
ptY = linspace(min(N2O_X_train(:,11)),max(N2O_X_train(:,11)),numPoints)';
[pd,x,y] = partialDependence(RFModel_croplands,{'MAT','NO3'},'QueryPoints',[ptX ptY]);

t = tiledlayout(5,5,'TileSpacing','compact');
ax1 = nexttile(1,[5,5]);
imagesc(x,y,pd),title('Partial Dependence Plot'),colorbar('eastoutside'),ax1.YDir = 'normal';
xlabel('MAT'),ylabel('NO3')
print(gcf,'Partial dependce_croplands_MATNO3','-dpng','-r600')

%
numPoints = 10;
ptX = linspace(min(N2O_X_train(:,2)),max(N2O_X_train(:,2)),numPoints)';
ptY = linspace(min(N2O_X_train(:,11)),max(N2O_X_train(:,11)),numPoints)';
[pd,x,y] = partialDependence(RFModel_croplands,{'MAT','NO3'},'QueryPoints',[ptX ptY]);

t = tiledlayout(5,5,'TileSpacing','compact');
ax1 = nexttile(1,[5,5]);
imagesc(x,y,pd),title('Partial Dependence Plot'),colorbar('eastoutside'),ax1.YDir = 'normal';
xlabel('MAP'),ylabel('NO3')
print(gcf,'Partial dependce_croplands_MAPNO3','-dpng','-r600')

%% 模型总体表现
figure()
plotregression(N2O_Y_test_global,RFPredict_global)
xlabel('Observerd N2O'),ylabel('Predict N2O')
legend('Random Forest','test data','Global')
hold off
print(gcf,'Regression_global','-dpng','-r600')

% Accuracy of RF    
RFRMSE=sqrt(sum(sum((RFPredict_global-N2O_Y_test_global).^2))/size(N2O_Y_test_global,1))
RFrMatrix=corrcoef(RFPredict_global,N2O_Y_test_global);
RFR2 = 1-(sumsqr(N2O_Y_test_global-RFPredict_global)/sumsqr(N2O_Y_test_global-mean(N2O_Y_test_global)))

%% 偏向关分析图
subplot(3,4,1)
plotPartialDependence(RFModel_forest,1)
hold on
plotPartialDependence(RFModel_grassland,1)
plotPartialDependence(RFModel_croplands,1)
xlabel('MAT'),ylabel(''),title(' ')
ax = gca
set(findobj(get(gca,'Children'),'LineWidth',0.5),'LineWidth',2);
hold off

subplot(3,4,2)
plotPartialDependence(RFModel_forest,2)
hold on
plotPartialDependence(RFModel_grassland,2)
plotPartialDependence(RFModel_croplands,2)
xlabel('MAP'),ylabel(''),title(' ')
ax = gca
set(findobj(get(gca,'Children'),'LineWidth',0.5),'LineWidth',2);
hold off

subplot(3,4,3)
plotPartialDependence(RFModel_forest,3)
hold on
plotPartialDependence(RFModel_grassland,3)
plotPartialDependence(RFModel_croplands,3)
xlabel('BD'),ylabel(''),title(' ')
ax = gca
set(findobj(get(gca,'Children'),'LineWidth',0.5),'LineWidth',2);
hold off

subplot(3,4,4)
plotPartialDependence(RFModel_forest,4)
hold on
plotPartialDependence(RFModel_grassland,4)
plotPartialDependence(RFModel_croplands,4)
xlabel('pH'),ylabel(''),title(' ')
ax = gca
set(findobj(get(gca,'Children'),'LineWidth',0.5),'LineWidth',2);
hold off

subplot(3,4,5)
plotPartialDependence(RFModel_forest,5)
hold on
plotPartialDependence(RFModel_grassland,5)
plotPartialDependence(RFModel_croplands,5)
xlabel('SOC'),ylabel(''),title(' ')
ax = gca
set(findobj(get(gca,'Children'),'LineWidth',0.5),'LineWidth',2);
hold off

subplot(3,4,6)
plotPartialDependence(RFModel_forest,6)
hold on
plotPartialDependence(RFModel_grassland,6)
plotPartialDependence(RFModel_croplands,6)
xlabel('TN'),ylabel(''),title(' ')
ax = gca
set(findobj(get(gca,'Children'),'LineWidth',0.5),'LineWidth',2);
hold off

subplot(3,4,7)
plotPartialDependence(RFModel_forest,7)
hold on
plotPartialDependence(RFModel_grassland,7)
plotPartialDependence(RFModel_croplands,7)
xlabel('TP'),ylabel(''),title(' ')
ax = gca
set(findobj(get(gca,'Children'),'LineWidth',0.5),'LineWidth',2);
hold off

subplot(3,4,8)
plotPartialDependence(RFModel_forest,8)
hold on
plotPartialDependence(RFModel_grassland,8)
plotPartialDependence(RFModel_croplands,8)
xlabel('SWC'),ylabel(''),title(' ')
ax = gca
set(findobj(get(gca,'Children'),'LineWidth',0.5),'LineWidth',2);
hold off

subplot(3,4,9)
plotPartialDependence(RFModel_forest,9)
hold on
plotPartialDependence(RFModel_grassland,9)
plotPartialDependence(RFModel_croplands,9)
xlabel('MBC'),ylabel(''),title(' ')
ax = gca
set(findobj(get(gca,'Children'),'LineWidth',0.5),'LineWidth',2);
hold off

subplot(3,4,10)
plotPartialDependence(RFModel_forest,10)
hold on
plotPartialDependence(RFModel_grassland,10)
plotPartialDependence(RFModel_croplands,10)
xlabel('MBN'),ylabel(''),title(' ')
ax = gca
set(findobj(get(gca,'Children'),'LineWidth',0.5),'LineWidth',2);
hold off

subplot(3,4,11)
plotPartialDependence(RFModel_forest,11)
hold on
plotPartialDependence(RFModel_grassland,11)
plotPartialDependence(RFModel_croplands,11)
xlabel('NO3'),ylabel(''),title(' ')
ax = gca
set(findobj(get(gca,'Children'),'LineWidth',0.5),'LineWidth',2);
hold off

subplot(3,4,12)
plotPartialDependence(RFModel_forest,12)
hold on
plotPartialDependence(RFModel_grassland,12)
plotPartialDependence(RFModel_croplands,12)
xlabel('NH4'),ylabel(''),title(' ')
ax = gca
set(findobj(get(gca,'Children'),'LineWidth',0.5),'LineWidth',2);
hold off

% print -djpeg -r600 N2O-Partial-dependce
print(gcf,'N2O-Partial dependce','-dpng','-r600')