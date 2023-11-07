% -*- coding: GBK -*-
% Created on Nov 2 2022 by Jiaqiang Liao
% 调试模型预测模块，预处理全球数据产品，为N2O_04_Final_anlysis做准备
clc,clear all

%% 1.Read predict database 全球数据产品读取
% land cover data
Input_way1 ='D:\研究生学习\氮循环\N N2O\data\Modis land cover-2016\MCD12C1.A2020001.061.2022172062638.hdf';    
		%hdf文件在电脑中的保存位置
Input_way2 ='/MOD12C1/Data Fields/Majority_Land_Cover_Type_1'; 
		%第一种分类数据在hdf文件中的位置         
Landcover = hdfread(Input_way1,Input_way2, 'Index', {[1  1],[1  1],[3600  7200]});

% 绘图浏览
% 将数据中的空值（值为255）改为17，方便设置颜色图
Landcover(Landcover==255) = 17;
caxis([0 17]);          %设置显示颜色范围
mycolor=[           	%设置颜色图
    0.4 0.4 0.4;            % 0     Water*
    0 98/255 65/255;        % 1     Evergreen Needleleaf forest	
    72/255 150/255 32/255;  % 2     Evergreen Broadleaf forest	
    0 160/255 107/255;      % 3     Deciduous Needleleaf forest	
    91/255 189/255 43/255;  % 4     Deciduous Broadleaf forest	
	131/255 199/255 93/255; % 5     Mixed forest	
    0 132/255 137/255;      % 6     Closed shrublands	
    110/255 195/255 201/255;% 7     Open shrublands	
    156/255 153/255 0;      % 8	    Woody savannas	
    252/255 245/255 78/255; % 9	    Savannas	
    243/255 194/255 70/255; % 10	Grasslands	
    160/255 149/255 196/255;% 11	Permanent*
    189/255 107/255 9/255;  % 12	Croplands	
    139/255 0 22/255;       % 13	Urban and built-up	
    236/255 135/255 14/255; % 14	Cropland/Natural vegetation mosaic	 
    115/255 136/255 193/255;% 15	Snow and ice*
    170/255 135/255 184/255;% 16	Barren or sparsely vegetated	
    1 1 1;                  % 17	Fill Value/Unclassified*
    ];     
% 使用设置好的颜色图绘图
figure()
    colormap(mycolor);
    y1=[0,3600];
    x1=[0,7200];
    axis([0 7200 0 3600]);
    image(x1,y1,Landcover);
    daspect([1 1 1]);
% 设置坐标轴
    xticks([1 900 1800 2700 3600 4500 5400 6300 7200])
    xticklabels({'180°','135°','90°','45°','0°','45°','90°','135°','180°'})
    yticks([1 900 1800 2700 3600])
    yticklabels({'90°','45°','0°','45°','90°'})
    title('全球土地覆盖类型图')
% 设置colorbar
 colorbar('Ticks',[0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5,...
		   12.5,13.5,14.5,15.5,16.5,17.5],...
          'TickLabels',{'水体','常绿针叶林','常绿阔叶林','落叶针叶林',...
          '落叶阔叶林','混交林','密集灌木','疏松灌木',...
          '多树荒原','荒原','草原','永久湿地','农田','城市与建成区',...
          '农田与天然植被相交','冰与雪','贫瘠','无数据或未分类'})

% Annual Mean Temperature = BIO1-Cliamte data (2.0°×2.5°)    (1970-2000)
datapath = 'D:\研究生学习\氮循环\N N2O\data\WorldClim-1970-2000\wc2.1_10m_bio\';
input = strcat(datapath,'wc2.1_10m_bio_1.tif');
MAT = imread(input); 

% Annual Precipitation = BIO12-Cliamte data (2.0°×2.5°)   
datapath = 'D:\研究生学习\氮循环\N N2O\data\WorldClim-1970-2000\wc2.1_10m_bio\';
input = strcat(datapath,'wc2.1_10m_bio_12.tif');
MAP = imread(input); 

% Temperature Seasonality (standard deviation ×100) = BIO4-Cliamte data (2.0°×2.5°)   
datapath = 'D:\研究生学习\氮循环\N N2O\data\WorldClim-1970-2000\wc2.1_10m_bio\';
input = strcat(datapath,'wc2.1_10m_bio_4.tif');
Temp_s = imread(input); 

% Precipitation Seasonality (Coefficient of Variation) = BIO15-Cliamte data (2.0°×2.5°)   
datapath = 'D:\研究生学习\氮循环\N N2O\data\WorldClim-1970-2000\wc2.1_10m_bio\';
input = strcat(datapath,'wc2.1_10m_bio_15.tif');
Prep_s = imread(input); 

% Bulk density- g/cm3 -(Scale factor:0.01)
datapath = 'D:\研究生学习\氮循环\N N2O\data\soil data- DaiYJ-2014\';
input = strcat(datapath,'BD1.tif');
BD = imread(input);

% pH (Scale factor:0.1)
datapath = 'D:\研究生学习\氮循环\N N2O\data\soil data- DaiYJ-2014\';
input = strcat(datapath,'PHH2O1.tif');
pH = imread(input);

% SOC- % of weight -(Scale factor:0.01)
datapath = 'D:\研究生学习\氮循环\N N2O\data\soil data- DaiYJ-2014\';
input = strcat(datapath,'SOC.tif');
SOC = imread(input);

% TN- % of weight -(Scale factor:0.01)
datapath = 'D:\研究生学习\氮循环\N N2O\data\soil data- DaiYJ-2014\';
input = strcat(datapath,'TN.tif');
TN =imread(input);

% TP- % of weight -(Scale factore:0.0001)
datapath = 'D:\研究生学习\氮循环\N N2O\data\soil data- DaiYJ-2014\';
input = strcat(datapath,'TP1.tif');
TP = imread(input);

% SWC- % of volume
datapath = 'D:\研究生学习\氮循环\N N2O\data\soil data- DaiYJ-2014\';
input = strcat(datapath,'SWC1.tif'); %应该是体积含水量
SWC = imread(input);

% Micro biomass carbon and nitrogen - g/m2 (1970-2012)
datapath = 'D:\研究生学习\氮循环\N N2O\data\Microbial biomass CNP-2012\data\';
input = strcat(datapath,'Global_Soil_Microbial_BiomassCN.nc');
ncdisp(input,'/','full')
% MBC-depth30cm- g/m2
MBC = ncread(input,'SMC30cm');
% MBN-depth30cm- g/m2
MBN = ncread(input,'SMN30cm');

% NH4+_N deposition data(0.5°，360*720) - g N m-2 mon-1
datapath = 'D:\研究生学习\氮循环\N N2O\data\N deposition-ISIMIP3b_0.5°\';
input = strcat(datapath,'ndep-nhx_2015soc_monthly_1850_2014.nc');
ncdisp(input,'/','full')         %浏览nc文件数据情况
nhx = ncread(input,'nhx');       %读取nc数据
nhx = nhx(:,:,1969:1980);        %读取2014年12个月
nhx = sum(nhx,3);                %累加求得2014年总沉降

% NO3-_N deposition data(0.5°，360*720) - g N m-2 mon-1
datapath = 'D:\研究生学习\氮循环\N N2O\data\N deposition-ISIMIP3b_0.5°\';
input = strcat(datapath,'ndep-noy_2015soc_monthly_1850_2014.nc');
ncdisp(input,'/','full')         %浏览nc文件数据情况
noy = ncread(input,'noy');
noy = noy(:,:,1969:1980);   
noy = sum(noy,3);

% Global Fertilizer and Manure dataset (0.5°，279*720) 1994-2001 - kg ha-1
datapath = "D:\研究生学习\氮循环\N N2O\写作\STOEN返修补充分析\NFertilizer_global_geotif\";
input = strcat(datapath,'nfertilizer_global.tif');
Nfer = imread(input);


%% 2.Data preprocessing 数据预处理
%包括数据读入，缺失值处理，尺度转换，单位换算，排列格式整理等

% land cover data
Landcover = imresize(Landcover,[360,720],'nearest');%统一图像尺寸,采用最近邻插值算法
Landcover = reshape(Landcover,259200,1);

% Annual Mean Temperature
MAT = single(MAT);   
MAT(MAT < -32767 | MAT > 100)= nan;        % set invalid values to nan
MAT = imresize(MAT,[360,720],'nearest');   %统一图像尺寸,采用双立方插值算法
figure,imshow(MAT)  %检查数据
title('MAT')
MAT = reshape(MAT,259200,1);
    
% Annual Precipitation
MAP = single(MAP);
MAP(MAP < -32767 | MAP <0)= 0; 
MAP = imresize(MAP,[360,720],'nearest');
figure,imshow(MAP)  
title('MAP')
MAP = reshape(MAP,259200,1);

% Temperature Seasonality
Temp_s = single(Temp_s);
Temp_s(Temp_s < -32767 | Temp_s <0)= 0; 
Temp_s = imresize(Temp_s,[360,720],'nearest');
figure,imshow(Temp_s)  
title('Temp_s')
Temp_s = reshape(Temp_s,259200,1);

% Precipitation Seasonality
Prep_s = single(Prep_s);
Prep_s(Prep_s < -32767 | Prep_s <0)= 0; 
Prep_s = imresize(Prep_s,[360,720],'nearest');
figure,imshow(Prep_s)  
title('Prep_s')
Prep_s = reshape(Prep_s,259200,1);


% Bulk density
BD = single(BD);                             
BD(BD  <= 0 ) = 0;   
BD = BD*0.01;                    %转换系数0.01

BDn = ones(706,43200)*nan;  %补充缺失区域
BDs = ones(4078,43200)*nan; %补充缺失区域
BD = [BDn;BD;BDs];

BD = imresize(BD,[360,720],'nearest');
figure,imshow(BD)  
title('BD')
BD = reshape(BD,259200,1);

% pH
pH = single(pH);
pH(pH >= 156) = 0;                        
pH = pH*0.1;                    %转换系数0.1

pHn = ones(706,43200)*nan;  %补充缺失区域
pHs = ones(4078,43200)*nan; %补充缺失区域
pH = [pHn;pH;pHs];

pH = imresize(pH,[360,720],'nearest');
figure,imshow(pH)  
title('pH')
pH = reshape(pH,259200,1);

% SOC
SOC = single(SOC);
SOC(SOC  <= 0 ) = 0; 
SOC = SOC*0.01;             %转换系数0.01
SOC = SOC*10;   %单位换算：% of weight 换算成 g*kg-1
SOC = log(SOC); %取对数

SOCn = ones(706,43200)*nan;  %补充缺失区域
SOCs = ones(4078,43200)*nan; %补充缺失区域
SOC = [SOCn;SOC;SOCs];

SOC = imresize(SOC,[360,720],'nearest');
figure,imshow(SOC)  
title('SOC')
SOC = reshape(SOC,259200,1);

% TN
TN = single(TN);
TN(TN  <= 0 ) = 0; 
TN = TN*0.01 ;                %转换系数0.01
TN = TN*10;   %单位换算：% of weight 换算成 g*kg-1
TN = log(TN); %取对数

TNn = ones(706,43200)*nan;  %补充缺失区域
TNs = ones(4078,43200)*nan; %补充缺失区域
TN = [TNn;TN;TNs];

TN = imresize(TN,[360,720],'nearest');
figure,imshow(TN)  
title('TN')
TN = reshape(TN,259200,1);

% TP
TP = single(TP);
TP(TP <= 0 ) = 0;       
TP = TP*0.0001;            %转换系数0.0001
TP = TP*10000;   %单位换算：% of weight 换算成 mg*kg-1
TP = log(TP);

TPn = ones(706,43200)*nan;  %补充缺失区域
TPs = ones(4078,43200)*nan; %补充缺失区域
TP = [TPn;TP;TPs];

TP = imresize(TP,[360,720],'nearest');
figure,imshow(TP) 
title('TP')
TP = reshape(TP,259200,1);

% SWC
SWC = single(SWC);
SWC(SWC <= 0 ) = 0;     

SWCn = ones(706,43200)*nan;  %补充缺失区域
SWCs = ones(4078,43200)*nan; %补充缺失区域
SWC = [SWCn;SWC;SWCs];

SWC = imresize(SWC,[360,720],'nearest');
figure,imshow(SWC)  
title('SWC')
SWC = reshape(SWC,259200,1);

% MBC
MBC = MBC';                     %转置
MBC = flipud(MBC);              %上下翻转
MBC = single(MBC);
MBC(MBC  <= 0 ) = 0;
MBC = log(MBC);
figure,imshow(MBC)
title('MBC')
MBC = reshape(MBC,259200,1);

% MBN
MBN = MBN';                     %转置
MBN = flipud(MBN);              %上下翻转
MBN = single(MBN);
MBN(MBN  <= 0 ) = 0;
MBN = log(MBN);
figure,imshow(MBN)
title('MBN')
MBN = reshape(MBN,259200,1);

% NH4+ deposition -g N m-2 yr-1
nhx = single(nhx');
nhx( nhx <0.01 )= 0;
nhx = log(nhx);
figure,imshow(nhx)  
title('NH4+')
nhx = reshape(nhx,259200,1);

% NO3- deposition -g N m-2 yr-1
noy = single(noy');
noy( noy <0.01 )= 0;
noy = log(noy);
figure,imshow(noy) 
title('NO3-')
noy = reshape(noy,259200,1);

% N fertilizer rate - kg ha-1
Nfer = single(Nfer);
Nfer(Nfer <= 0 ) = nan;   
Nfern = ones(13,720)*nan;  %补充缺失区域
Nfers = ones(68,720)*nan; %补充缺失区域
Nfer = [Nfern;Nfer;Nfers];
Nfer = imresize(Nfer,[360,720],'nearest');
figure,imshow(Nfer)  
title('Nfer')
Nfer = reshape(Nfer,259200,1);

N2O_predict = single(0*[1:259200]'); %创建空数据集
X_predict = [MAT';MAP';BD';pH';SOC';TN';TP';SWC';MBC';MBN';noy';nhx';Nfer';Temp_s';Prep_s']';
save X_predict X_predict

%% 3.Estimation of global N2O 
% 利用数据产品和训练后的模型进行全球格局预测
%[trainedModel,validationRMSE,validationR2] = trainRegressionModel_RF(N2O_X,N2O_Y);%构建于N2O_01
clear all,clc

% land cover data
Input_way1 ='D:\研究生学习\氮循环\N N2O\data\Modis land cover-2016\MCD12C1.A2020001.061.2022172062638.hdf';    
		%hdf文件在电脑中的保存位置
Input_way2 ='/MOD12C1/Data Fields/Majority_Land_Cover_Type_1'; 
		%第一种分类数据在hdf文件中的位置         
Landcover = hdfread(Input_way1,Input_way2, 'Index', {[1  1],[1  1],[3600  7200]});

Landcover = imresize(Landcover,[360,720],'nearest');%统一图像尺寸,采用最近邻插值算法
Landcover = reshape(Landcover,259200,1);

load X_predict.mat  %读取预测数据

% 选择出陆地区域进行模拟
X1 = X_predict(Landcover >= 1 & Landcover <= 17,:);

load RFmodel_N2O.mat
N2O_predict(Landcover >= 1 & Landcover <= 17) = predict(RFModel,X1); %构建于N2O_02
N2O_predict(N2O_predict<0)=0.0;
N2O_predict(Landcover  == 0) = nan;  %水体剔除
N2O_predict(Landcover >= 6 & Landcover <= 7) = nan;
N2O_predict(Landcover == 11) = nan;  %多年冻土剔除
N2O_predict(Landcover >=15) = nan;  %冻土、城市、未分类

save N2O_predict N2O_predict
N2O_predict = reshape(N2O_predict,[360,720]);

N2O_blank = single(nan*[1:259200]');
N2O_blank(Landcover >= 6 & Landcover <= 7) = 0;
N2O_blank(Landcover == 11) = 0;
N2O_blank(Landcover >=16) = 0;
N2O_blank = reshape(N2O_blank,[360,720]);

% 保存图
figure1 = figure
axes1 = axes('Parent',figure1);
hold(axes1,'on');
surf(N2O_blank','EdgeColor','#BEBEBE');
mesh(N2O_predict','Parent',axes1);
title({'Random forest'});   
xlim(axes1,[0 360]);
ylim(axes1,[0 720]);
view(axes1,[89.9875 90]);
grid(axes1,'on');
hold(axes1,'off');
% 设置其余坐标区属性
set(axes1,'XTick',[0 60 120 180 240 300 360],'XTickLabel',...
    {'90°N','60°N','30°N','0°','30°S','60°S','90°S'},'YTick',...
    [120 240 360 480 600],'YTickLabel',{'120°W','60°W','0°','60°E','120°E'});
% 创建 colorbar
S = load('spine.mat');
colormap(brewermap([],"YlGnBu"))
% colormap (axes1,flipud(bone))
colorbar(axes1);
caxis([3.5 7.5]);

% saveas(gcf,'D:\研究生学习\氮循环\N N2O\写作\一稿代码和图\N2O-Map.jpg')

%% 4.Estimation of each lancover N2O
clc,clear all
% land cover data
Input_way1 ='D:\研究生学习\氮循环\N N2O\data\Modis land cover-2016\MCD12C1.A2020001.061.2022172062638.hdf';    
		%hdf文件在电脑中的保存位置
Input_way2 ='/MOD12C1/Data Fields/Majority_Land_Cover_Type_1'; 
		%第一种分类数据在hdf文件中的位置         
Landcover = hdfread(Input_way1,Input_way2, 'Index', {[1  1],[1  1],[3600  7200]});

Landcover = imresize(Landcover,[360,720],'nearest');%统一图像尺寸,采用最近邻插值算法
Landcover = reshape(Landcover,259200,1);

load X_predict.mat  %读取预测数据

%% 4.1 forest N2O
load RFmodel_N2O_forest.mat
N2O_predict_forest = single(0*[1:259200]'); %创建空数据集
X1 = X_predict(Landcover >=1 & Landcover <=5,:);
N2O_predict_forest(Landcover >=1 & Landcover <=5,:) = predict(RFModel_forest,X1); 
N2O_predict_forest(N2O_predict_forest<0)=0.0;
N2O_predict_forest(Landcover == 0) = nan; 
N2O_predict_forest(Landcover >= 6 | Landcover <= 17) = nan; 
N2O_predict_forest = reshape(N2O_predict_forest,[360,720]);

figure1 = figure
axes1 = axes('Parent',figure1);
hold(axes1,'on');
% 创建 mesh
mesh(N2O_predict_forest','Parent',axes1);
title({'Forest N2O'});   
xlim(axes1,[0 360]);
ylim(axes1,[0 720]);
view(axes1,[89.9875 90]);
grid(axes1,'on');
hold(axes1,'off');
% 设置其余坐标区属性
set(axes1,'XTick',[0 60 120 180 240 300 360],'XTickLabel',...
    {'90°N','60°N','30°N','0°','30°S','60°S','90°S'},'YTick',...
    [120 240 360 480 600],'YTickLabel',{'120°W','60°W','0°','60°E','120°E'});
% 创建 colorbar
S = load('spine.mat');
colormap(brewermap([],"YlGnBu"))
% colormap (axes1,flipud(bone))
colorbar(axes1);
caxis([3.5 7.5]);

% saveas(gcf,'D:\研究生学习\氮循环\N N2O\写作\一稿代码和图\N2O_forest-Map.jpg')

%% 4.2 grassland N2O
load RFmodel_N2O_grassland.mat
N2O_predict_grassland = single(0*[1:259200]'); %创建空数据集
X1 = X_predict( Landcover >= 8 | Landcover <= 10,:);
N2O_predict_grassland(Landcover >= 8 | Landcover <= 10) = predict(RFModel_grassland,X1); %构建于N2O_02
N2O_predict_grassland(N2O_predict_grassland<0)=0.0;
N2O_predict_grassland(Landcover >= 0 & Landcover <= 7) = nan; 
N2O_predict_grassland(Landcover >= 11 & Landcover <= 17) = nan; 
N2O_predict_grassland = reshape(N2O_predict_grassland,[360,720]);

figure1 = figure
axes1 = axes('Parent',figure1);
hold(axes1,'on');
% 创建 mesh
mesh(N2O_predict_grassland','Parent',axes1);
title({'Grassland N2O'});   
xlim(axes1,[0 360]);
ylim(axes1,[0 720]);
view(axes1,[89.9875 90]);
grid(axes1,'on');
hold(axes1,'off');
% 设置其余坐标区属性
set(axes1,'XTick',[0 60 120 180 240 300 360],'XTickLabel',...
    {'90°N','60°N','30°N','0°','30°S','60°S','90°S'},'YTick',...
    [120 240 360 480 600],'YTickLabel',{'120°W','60°W','0°','60°E','120°E'});
% 创建 colorbar
S = load('spine.mat');
colormap(brewermap([],"YlGnBu"))
% colormap (axes1,flipud(bone))
colorbar(axes1);
caxis([3 7.5]);

% saveas(gcf,'D:\研究生学习\氮循环\N N2O\写作\一稿代码和图\N2O_grassland-Map.jpg')

%% 4.3 cropland N2O
load RFmodel_N2O_croplands.mat
N2O_predict_croplands = single(0*[1:259200]'); %创建空数据集
X1 = X_predict(Landcover >= 12 & Landcover <= 14,:);
N2O_predict_croplands(Landcover >= 12 & Landcover <= 14,:) = predict(RFModel_croplands,X1); %构建于N2O_02
N2O_predict_croplands(N2O_predict_croplands<0)=0.0;
N2O_predict_croplands(Landcover >= 0 & Landcover <= 11) = nan; 
N2O_predict_croplands(Landcover >= 15 & Landcover <= 17) = nan; 
N2O_predict_croplands = reshape(N2O_predict_croplands,[360,720]);

figure1 = figure
axes1 = axes('Parent',figure1);
hold(axes1,'on');
% 创建 mesh
mesh(N2O_predict_croplands','Parent',axes1);
title({'Croplands N2O'});   
xlim(axes1,[0 360]);
ylim(axes1,[0 720]);
view(axes1,[89.9875 90]);
grid(axes1,'on');
hold(axes1,'off');
% 设置其余坐标区属性
set(axes1,'XTick',[0 60 120 180 240 300 360],'XTickLabel',...
    {'90°N','60°N','30°N','0°','30°S','60°S','90°S'},'YTick',...
    [120 240 360 480 600],'YTickLabel',{'120°W','60°W','0°','60°E','120°E'});
% 创建 colorbar
S = load('spine.mat');
colormap(brewermap([],"YlGnBu"))
% colormap (axes1,flipud(bone))
colorbar(axes1);
caxis([3 7.5]);

% saveas(gcf,'D:\研究生学习\氮循环\N N2O\写作\一稿代码和图\N2O_croplands-Map.jpg')


%% 4.4 combined all type N2O
N2O_predict_nature = single(nan*[1:259200]');
N2O_predict_nature(Landcover >=1 & Landcover <=5) = N2O_predict_forest(Landcover >=1 & Landcover <=5);
N2O_predict_nature(Landcover >= 8 & Landcover <= 10) = N2O_predict_grassland(Landcover >= 8 & Landcover <= 10);

N2O_predict2 = single(nan*[1:259200]'); %创建空数据集
N2O_predict2(Landcover >=1 & Landcover <=5) = N2O_predict_forest(Landcover >=1 & Landcover <=5);
N2O_predict2(Landcover >= 8 & Landcover <= 10) = N2O_predict_grassland(Landcover >= 8 & Landcover <= 10);
N2O_predict2(Landcover >= 12 & Landcover <= 14) = N2O_predict_croplands(Landcover >= 12 & Landcover <= 14);
save N2O_predict2 N2O_predict2
N2O_predict2 = reshape(N2O_predict2,[360,720]);

N2O_blank = single(nan*[1:259200]');
N2O_blank(Landcover >= 6 & Landcover <= 7) = 0;
N2O_blank(Landcover == 11) = 0;
N2O_blank(Landcover >=16) = 0;
N2O_blank = reshape(N2O_blank,[360,720]);

figure1 = figure
axes1 = axes('Parent',figure1);
hold(axes1,'on');
surf(N2O_blank','EdgeColor','#BEBEBE');
mesh(N2O_predict2','Parent',axes1);
title({'N2O map-combined'});   
xlim(axes1,[0 360]);
ylim(axes1,[0 720]);
view(axes1,[89.9875 90]);
grid(axes1,'on');
hold(axes1,'off');
% 设置其余坐标区属性
set(axes1,'XTick',[0 60 120 180 240 300 360],'XTickLabel',...
    {'90°N','60°N','30°N','0°','30°S','60°S','90°S'},'YTick',...
    [120 240 360 480 600],'YTickLabel',{'120°W','60°W','0°','60°E','120°E'});
% 创建 colorbar
S = load('spine.mat');
colormap(brewermap([],"YlGnBu"))
% colormap (axes1,flipud(bone))
colorbar(axes1);
caxis([3 7]);

% saveas(gcf,'D:\研究生学习\氮循环\N N2O\写作\一稿代码和图\N2O-Map-combined.jpg')
%print(gcf,'N2O_Map-combined','-dpng','-r1200')
