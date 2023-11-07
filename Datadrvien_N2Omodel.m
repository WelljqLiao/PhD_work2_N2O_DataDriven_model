% -*- coding: GBK -*-
% Created on Juny 9 2023 by Jiaqiang Liao
% This code is used to estimate the global N2O emission rates
% The input data includes global observation datasets (N2O_database.mat) and forecast datasets (X_predict.mat)
% See Methods and Supplementary Materials in the paper for details
clc,clear all

% load N2O field database
load N2O_database.mat 

% Weighted sampling based on observation time to create a sample data set
[n, D] = size(N2O);
ID = 1:n;
% rng(200)
samples = randsample(ID, n, true, weights); %The weight is determined according to the observation time
N2O = N2O(ID,:);
N2O_X = N2O(:,1:13); % 'MAT','MAP','BD','pH','SOC','TN','TP','SM','MBC','MBN','NO3','NH4','Nfer'
N2O_Y = N2O(:,16); % N2O emission rate

%% ------ 1 Random Forest Model ------ %%
% ------ 1.1 forest(n=679) ------ %
N2O_X_forest = N2O_X(LandID == 3,:); % The LandID of the forest is 3
N2O_Y_forest = N2O_Y(LandID == 3,:);
Input = N2O_X_forest;
Output = N2O_Y_forest;
% rng(200)
[ndata, D] = size(N2O_X_forest);          
R = randperm(ndata,round(0.2*ndata));  % test data
N2O_X_test = N2O_X_forest(R,:);     
N2O_Y_test = N2O_Y_forest(R,:);
N2O_X_forest(R,:) = []; % train data
N2O_Y_forest(R,:) = [];
N2O_X_train = N2O_X_forest;          
N2O_Y_train = N2O_Y_forest;
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
if RFRMSE<1.16     
    disp(RFRMSE);
    break;
end
end
forest_R2 = 1-(sumsqr(N2O_Y_test-RFPredictYield)/sumsqr(N2O_Y_test-mean(N2O_Y_test)))
plotregression(N2O_Y_test,RFPredictYield)

% Model Performance Graph
figure()
plot(log(exp(N2O_Y_test)*365*0.000001),log(exp(RFPredictYield)*365*0.000001),'.','MarkerSize',15);
axis([-6 2 -6 2]);
set(gcf,'position',[100,100,500,450]);
line([-6:2],[-6:2],'color','k')
xlabel('Observerd, ln(N_2O,g m^-^2 year^-^1)'),ylabel('Predict, ln(N_2O,g m^-^2 year^-^1)')
text(-5.5,1,'Fitting slope = 0.7204','FontSize',10)
text(-5.5,1.5,'Ecosystem type: forest','FontSize',10)

% predict imformation
RFPredict_global = RFPredictYield; 
N2O_Y_test_global = N2O_Y_test;  
RI_forest = RFModel_forest.OOBPermutedPredictorDeltaError/max(RFModel_forest.OOBPermutedPredictorDeltaError);

% ------ 1.2 grassland(n=514)  ------ %
N2O_X_grassland = N2O_X(LandID == 4,:);
N2O_Y_grassland = N2O_Y(LandID == 4,:);
Input = N2O_X_grassland;
Output = N2O_Y_grassland;
% rng(200)
[ndata, D] = size(N2O_X_grassland);         
R = randperm(ndata,round(0.2*ndata));      
N2O_X_test = N2O_X_grassland(R,:);     
N2O_Y_test = N2O_Y_grassland(R,:);
N2O_X_grassland(R,:) = [];
N2O_Y_grassland(R,:) = [];
N2O_X_train = N2O_X_grassland;        
N2O_Y_train = N2O_Y_grassland;
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
if RFRMSE<1.2    
    disp(RFRMSE);
    break;
end
end
grassland_R2 = 1-(sumsqr(N2O_Y_test-RFPredictYield)/sumsqr(N2O_Y_test-mean(N2O_Y_test)))
plotregression(N2O_Y_test,RFPredictYield)

% Model Performance Graph
figure()
plot(log(exp(N2O_Y_test)*365*0.000001),log(exp(RFPredictYield)*365*0.000001),'.','MarkerSize',15);
axis([-6 2 -6 2]);
set(gcf,'position',[100,100,500,450]);
line([-6:2],[-6:2],'color','k')
xlabel('Observerd, ln(N_2O,g m^-^2 year^-^1)'),ylabel('Predict, ln(N_2O,g m^-^2 year^-^1)')
text(-5.5,1,'Fitting slope = 0.8307','FontSize',10)
text(-5.5,1.5,'Ecosystem type: grassland','FontSize',10)

% predict imformation
RFPredict_global = [RFPredict_global,
    RFPredictYield];  
N2O_Y_test_global = [N2O_Y_test_global,
    N2O_Y_test];     
RI_grassland = RFModel_grassland.OOBPermutedPredictorDeltaError/max(RFModel_grassland.OOBPermutedPredictorDeltaError);

% ------ 1.3 croplands(n=4356) ------ %
N2O_X_croplands = N2O_X(LandID == 1,:);
N2O_Y_croplands = N2O_Y(LandID == 1,:);
Input = N2O_X_croplands;
Output = N2O_Y_croplands;
% rng(200)
[ndata, D] = size(N2O_X_croplands);        
R = randperm(ndata,round(0.2*ndata));      
N2O_X_test = N2O_X_croplands(R,:);     
N2O_Y_test = N2O_Y_croplands(R,:);
N2O_X_croplands(R,:) = [];
N2O_Y_croplands(R,:) = [];
N2O_X_train = N2O_X_croplands;          
N2O_Y_train = N2O_Y_croplands;
RFRunNumSet=10;
for RFCycleRun=1:RFRunNumSet
nTree=100;nLeaf=5;
RFModel_croplands=TreeBagger(nTree,N2O_X_train,N2O_Y_train,...
    'Method','regression','OOBPredictorImportance','on', 'MinLeafSize',nLeaf, ...
    'PredictorNames',{'MAT','MAP','BD','pH','SOC','TN','TP','SM','MBC','MBN','NO3','NH4','Nfer'});
[RFPredictYield,RFPredictConfidenceInterval]=predict(RFModel_croplands,N2O_X_test);

% Accuracy of RF    
RFRMSE=sqrt(sum(sum((RFPredictYield-N2O_Y_test).^2))/size(N2O_Y_test,1));
if RFRMSE<1.1      
    disp(RFRMSE);
    break;
end
end
cropland_R2 = 1-(sumsqr(N2O_Y_test-RFPredictYield)/sumsqr(N2O_Y_test-mean(N2O_Y_test)))
plotregression(N2O_Y_test,RFPredictYield)

% Model Performance Graph
figure()
plot(log(exp(N2O_Y_test)*365*0.000001),log(exp(RFPredictYield)*365*0.000001),'.','MarkerSize',15);
axis([-6 2 -6 2]);
set(gcf,'position',[100,100,500,450]);
line([-6:2],[-6:2],'color','k')
xlabel('Observerd, ln(N_2O,g m^-^2 year^-^1)'),ylabel('Predict, ln(N_2O,g m^-^2 year^-^1)')
text(-5.5,1,'Fitting slope = 0.7954','FontSize',10)
text(-5.5,1.5,'Ecosystem type: cropland','FontSize',10)

% predict imformation
RFPredict_global = [RFPredict_global,
    RFPredictYield];  
N2O_Y_test_global = [N2O_Y_test_global,
    N2O_Y_test];     
RI_cropland = RFModel_croplands.OOBPermutedPredictorDeltaError/max(RFModel_croplands.OOBPermutedPredictorDeltaError);

% ------ 1.4 global ------ %
plotregression(N2O_Y_test_global,RFPredict_global)

% Model Performance Graph
figure()
plot(log(exp(N2O_Y_test_global)*365*0.000001),log(exp(RFPredict_global)*365*0.000001),'.','MarkerSize',15);
axis([-6 2 -6 2]);
set(gcf,'position',[100,100,500,450]);
line([-6:2],[-6:2],'color','k')
xlabel('Observerd, ln(N_2O,g m^-^2 year^-^1)'),ylabel('Predict, ln(N_2O,g m^-^2 year^-^1)')
text(-5.5,1,'Fitting slope = 0.7940','FontSize',10)
text(-5.5,1.5,'Ecosystem type: global','FontSize',10)

% Accuracy of global performence
Global_RFR2 = 1-(sumsqr(N2O_Y_test_global-RFPredict_global)/sumsqr(N2O_Y_test_global-mean(N2O_Y_test_global)))

%% ------ 2 Model train (100cycles) %%
clc,clear all
load N2O_database.mat 
for i = 1:1000
[n, D] = size(N2O);
ID = 1:n;
samples = randsample(ID, n, true, weights); 
N2O = N2O(ID,:);
N2O_X = N2O(:,1:13);
N2O_Y = N2O(:,16);

% ------ 2.1 forest ------ %
N2O_X_forest = N2O_X(LandID == 3,:);
N2O_Y_forest = N2O_Y(LandID == 3,:);
boot_id = (1:679);
boot_id = boot(boot_id,100);
Input = N2O_X_forest(boot_id(:,i),:);
Output = N2O_Y_forest(boot_id(:,i),:);
[ndata, D] = size(N2O_X_forest);          
R = randperm(ndata,round(0.2*ndata));     
N2O_X_test = N2O_X_forest(R,:);     
N2O_Y_test = N2O_Y_forest(R,:);
N2O_X_forest(R,:) = [];
N2O_Y_forest(R,:) = [];
N2O_X_train = N2O_X_forest;          
N2O_Y_train = N2O_Y_forest;
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
if RFRMSE<1.16     
    break;
end
end
% predict imformation
RFPredict_global = RFPredictYield;  
N2O_Y_test_global = N2O_Y_test;   
RI_forest = RFModel_forest.OOBPermutedPredictorDeltaError/max(RFModel_forest.OOBPermutedPredictorDeltaError);

% ------ 2.2 grassland ------ %
N2O_X_grassland = N2O_X(LandID == 4,:);
N2O_Y_grassland = N2O_Y(LandID == 4,:);
boot_id = (514:-1:1);  
boot_id = boot(boot_id,100);
Input = N2O_X_grassland(boot_id(:,i),:);
Output = N2O_Y_grassland(boot_id(:,i),:);
[ndata, D] = size(N2O_X_grassland);        
R = randperm(ndata,round(0.2*ndata));     
N2O_X_test = N2O_X_grassland(R,:);     
N2O_Y_test = N2O_Y_grassland(R,:);
N2O_X_grassland(R,:) = [];
N2O_Y_grassland(R,:) = [];
N2O_X_train = N2O_X_grassland;   
N2O_Y_train = N2O_Y_grassland;
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
if RFRMSE<1.2   
    break;
end
end
% predict imformation
RFPredict_global = [RFPredict_global,
    RFPredictYield]; 
N2O_Y_test_global = [N2O_Y_test_global,
    N2O_Y_test];   
RI_grassland = RFModel_grassland.OOBPermutedPredictorDeltaError/max(RFModel_grassland.OOBPermutedPredictorDeltaError);

% ------ 2.3 cropland ------ %
N2O_X_croplands = N2O_X(LandID == 1,:);
N2O_Y_croplands = N2O_Y(LandID == 1,:);
boot_id = (1:4356); 
boot_id = boot(boot_id,100);
Input = N2O_X_croplands(boot_id(:,i),:);
Output = N2O_Y_croplands(boot_id(:,i),:);
[ndata, D] = size(N2O_X_croplands);        
R = randperm(ndata,round(0.2*ndata));   
N2O_X_test = N2O_X_croplands(R,:);    
N2O_Y_test = N2O_Y_croplands(R,:);
N2O_X_croplands(R,:) = [];
N2O_Y_croplands(R,:) = [];
N2O_X_train = N2O_X_croplands;      
N2O_Y_train = N2O_Y_croplands;
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
if RFRMSE<1.1     
    break;
end
end
% predict imformation
RFPredict_global = [RFPredict_global,
    RFPredictYield]; 
N2O_Y_test_global = [N2O_Y_test_global,
    N2O_Y_test];    
RI_cropland = RFModel_croplands.OOBPermutedPredictorDeltaError/max(RFModel_croplands.OOBPermutedPredictorDeltaError);

% Accuracy of global performence
Global_RFR2 = 1-(sumsqr(N2O_Y_test_global-RFPredict_global)/sumsqr(N2O_Y_test_global-mean(N2O_Y_test_global)))

%% ------ 3 Model Predictions ------ %%
% ------ load forecast data ------ %
load X_predict.mat
load Landcover.mat
X_predict = X_predict(:,[1:13]); 

% forest
N2O_predict_forest = single(nan*[1:259200]'); 
N2O_predict_forest(Landcover2 >=1 & Landcover2 <=17) = 0;
X1 = X_predict(Landcover2 == 1 | Landcover2 == 2 | Landcover2 == 3 | Landcover2 == 4 | Landcover2 == 5 ...
     | Landcover2 == 8 | Landcover2 == 9,:);
N2O_predict_forest(Landcover2 == 1 | Landcover2 == 2 | Landcover2 == 3 | Landcover2 == 4 | Landcover2 == 5 ...
     | Landcover2 == 8 | Landcover2 == 9,:) = predict(RFModel_forest,X1); 
N2O_predict_forest(N2O_predict_forest<0)=0;

% grassland
N2O_predict_grassland = single(nan*[1:259200]'); 
N2O_predict_grassland(Landcover2 >=1 & Landcover2 <=17) = 0;
X1 = X_predict( Landcover2 == 10,:);
N2O_predict_grassland(Landcover2 == 10,:) = predict(RFModel_grassland,X1); 
N2O_predict_grassland(N2O_predict_grassland<0)=0;

% cropland 
N2O_predict_croplands = single(nan*[1:259200]');
N2O_predict_croplands(Landcover2 >=1 & Landcover2 <=17) = 0;
X1 = X_predict(Landcover2 >= 12 & Landcover2 <= 14,:);
N2O_predict_croplands(Landcover2 >= 12 & Landcover2 <= 14,:) = predict(RFModel_croplands,X1); 
N2O_predict_croplands(N2O_predict_croplands<0)=0;

% global
N2O_predict2 = single(nan*[1:259200]'); 
N2O_predict2(Landcover2 >=1 & Landcover2 <=17) = 0;
N2O_predict2(Landcover2 == 1 | Landcover2 == 2 | Landcover2 == 3 | Landcover2 == 4 | Landcover2 == 5 ...
     | Landcover2 == 8 | Landcover2 == 9) = N2O_predict_forest(Landcover2 == 1 | Landcover2 == 2 | Landcover2 == 3 | Landcover2 == 4 | Landcover2 == 5 ...
     | Landcover2 == 8 | Landcover2 == 9);
N2O_predict2(Landcover2 == 10) = N2O_predict_grassland(Landcover2 == 10);
N2O_predict2(Landcover2 >= 12 & Landcover2 <= 14) = N2O_predict_croplands(Landcover2 >= 12 & Landcover2 <= 14);
  
% save result
N2O_100(:,i) = N2O_predict2;  
N2O_forest_100(:,i) = N2O_predict_forest;
N2O_grassland_100(:,i) = N2O_predict_grassland;
N2O_cropland_100(:,i) = N2O_predict_croplands;

RI_forest_100(:,i) = RI_forest';
RI_grassland_100(:,i) = RI_grassland';
RI_cropland_100(:,i) = RI_cropland';

Global_RFR2_100(:,i) = Global_RFR2
end

% save relative important results
RI_forest_mean = mean(RI_forest_100,2); % Extract variable importance information
RI_grassland_mean = mean(RI_grassland_100,2);
RI_croplands_mean = mean(RI_cropland_100,2);

%% ------ 4 global pattern and uncertainty ------ %%
N2O_forest_mean = mean(N2O_forest_100,2,"omitnan");
N2O_grassland_mean = mean(N2O_grassland_100,2,"omitnan");
N2O_cropland_mean = mean(N2O_cropland_100,2,"omitnan");

N2O_mean = mean(N2O_100,2,"omitnan");
N2O_std = std(N2O_100,1,2,"omitnan");

N2O_100t = exp(N2O_100)*365*0.000001; %unit converted to g m-2 yr-1
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

% ------ 4.1 forest ------ %
figure()
N2O_forest_meant = exp(N2O_forest_mean)*365*0.000001;
N2O_forest = reshape(N2O_forest_meant,[360,720]); 
lat = [-89.5:0.5:90];
lon = [-179.5:0.5:180];
[lon, lat] = meshgrid(lon,lat);
ax1 = axesm('MapProjection','pcarree','MapLatLimit',[-90 90],'MapLonLimit',[-180 180],'Frame','on','Grid','off', ...
    'FontName','Times','FontSize',12,'FEdgeColor','none', ...
    'MLineLocation',90,'MLabelRound', 0, 'MeridianLabel','on',...
    'PLineLocation',45,'PLabelRound', 0,'ParallelLabel','on','MLabelParallel','south');
tightmap;
mycolorpoint=[[207 223 39];...
    [110 212 74];...
    [48 176 127];...
    [34 144 145];...
    [43 112 145];...
    [63 70 140];...
    [63 37 84];...
    [230 230 230]];
mycolorposition=[1 36 50 64 84 105 127 128];
mycolormap_r=interp1(mycolorposition,mycolorpoint(:,1),1:128,'linear','extrap');
mycolormap_g=interp1(mycolorposition,mycolorpoint(:,2),1:128,'linear','extrap');
mycolormap_b=interp1(mycolorposition,mycolorpoint(:,3),1:128,'linear','extrap');
mycolor=[mycolormap_r',mycolormap_g',mycolormap_b']/255;
mycolor=round(mycolor*10^4)/10^4;
cm1 = colormap(ax1, mycolor);
cm1 = flipud(cm1);
colormap(ax1, cm1)
caxis([0 350*365*0.000001]);
N2O_forestf = flipud(N2O_forest);
i1 = surfm(lat, lon, N2O_forestf);
h1 = colorbar('FontName', 'Times', 'FontSize', 10);

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

% ------ 4.2 grassland ------ %
figure()
N2O_grassland_meant = exp(N2O_grassland_mean)*365*0.000001;
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

% ------ 4.3 cropland------ %
figure()
N2O_cropland_meant = exp(N2O_cropland_mean)*365*0.000001;
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

% ------ 4.4 gloabal ------ %
figure()
N2O_meant = exp(N2O_mean)*365*0.000001;
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
caxis([0 350*365*0.000001]);
N2O_meanf = flipud(N2O_global);
i1 = surfm(lat, lon, N2O_meanf);
h1 = colorbar('FontName', 'Times', 'FontSize', 10);

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
N2O_uncertaintyf = flipud(N2O_uncertainty);
i1 = surfm(lat, lon, N2O_uncertaintyf);
h1 = colorbar('FontName', 'Times', 'FontSize', 10);

% Save tiff
% N2O_mean(N2O_mean <= 0 ) = nan;
% N2O_meant = exp(N2O_mean)*365*0.000001;
% N2O_global= reshape(N2O_meant,[360,720]); 
% N2O_meanf = flipud(N2O_global);
% R = georasterref('RasterSize', size(N2O_meanf), 'LatitudeLimits', [-90 90], 'LongitudeLimits', [-180 180]);
% geotiffwrite('Global_N2O.tif', N2O_meanf, R,'CoordRefSysCode',4326);  

