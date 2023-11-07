% -*- coding: GBK -*-
% Created on Nov 2 2022 by Jiaqiang Liao
% 对N2O_04_Final_anlysis结果进行相关的统计分析
clc,clear all
%% 1.生态区格局
load meanvalue_0830.mat
% 分区均值和100次sd，置信区间
N2O_forest_100(N2O_forest_100 == 0) = nan;
N2O_grassland_100(N2O_grassland_100 == 0) = nan;
N2O_cropland_100(N2O_cropland_100 == 0) = nan;
N2O_100(N2O_100 == 0) = nan;

N2O_global_mean1 = mean(exp(N2O_100),1,'omitnan');
N2O_global_mean2 = mean(N2O_global_mean1,'omitnan')
N2O_global_sd = std(N2O_global_mean1,'omitnan')  %模型SD
max(max(exp(N2O_100))),min(min(exp(N2O_100)))

N2O_forest_mean1 = mean(exp(N2O_forest_100),1,'omitnan');
N2O_forest_mean2 = mean(N2O_forest_mean1,'omitnan')
N2O_forest_sd = std(N2O_forest_mean1,'omitnan')  %模型SD
max(max(exp(N2O_forest_100))),min(min(exp(N2O_forest_100)))

N2O_grassland_mean1 = mean(exp(N2O_grassland_100),1,'omitnan');
N2O_grassland_mean2 = mean(N2O_grassland_mean1,'omitnan')
N2O_grassland_sd = std(N2O_grassland_mean1,'omitnan')  %模型SD
max(max(exp(N2O_grassland_100))),min(min(exp(N2O_grassland_100)))

N2O_cropland_mean1 = mean(exp(N2O_cropland_100),1,'omitnan');
N2O_cropland_mean2 = mean(N2O_cropland_mean1,'omitnan')
N2O_cropland_sd = std(N2O_cropland_mean1,'omitnan')  %模型SD
max(max(exp(N2O_cropland_100))),min(min(exp(N2O_cropland_100)))

%示例
max(max(N2O_100)),min(min(N2O_100))
N2O_global_mean1 = exp(mean(N2O_100,1,"omitnan"));
N2O_global_mean2 = mean(N2O_global_mean1,"omitnan")
N2O_global_sd = std(N2O_global_mean1,1,"omitnan")
numb = length(N2O_global_mean1);
%求置信区间，均值±标准差/根号n *1.96
N2O_global_CIu = N2O_global_mean2 + 1/96*N2O_global_sd/sqrt(numb);
N2O_global_CId = N2O_global_mean2 - 1/96*N2O_global_sd/sqrt(numb);

%% 2.纬度N2O均值格局
%global(试趋势线平均，五点平均或十点平均，用误差棒来
N2O_mean = mean(N2O_100,2,"omitnan");
N2O_mean= reshape(N2O_mean,[360,720]); 
N2O_global_90 = imresize(N2O_mean,[180,720],'nearest');
N2O_lat = mean(N2O_global_90,2,"omitnan"); 
lat = 89.5:-1:-89.5;
figure(),plot(N2O_lat,lat')
N2O_lat = [N2O_lat,lat'];
title('global');
mesh(N2O_mean)

%forest
N2O_forest_m = mean(N2O_forest_100,2,"omitnan");
N2O_forest_m = reshape(N2O_forest_m,[360,720]); 
N2O_forest_90 = imresize(N2O_forest_m,[180,720],'nearest');
N2O_lat = mean(N2O_forest_90,2,"omitnan"); 
lat = 89.5:-1:-89.5;
figure(),plot(N2O_lat,lat')
N2O_lat = [N2O_lat,lat'];
title('forest');

%grassland
N2O_grassland_m = mean(N2O_grassland_100,2,"omitnan");
N2O_grassland_m= reshape(N2O_grassland_m,[360,720]); 
N2O_grassland_90 = imresize(N2O_grassland_m,[180,720],'nearest');
N2O_lat = mean(N2O_grassland_90,2,"omitnan"); 
lat = 89.5:-1:-89.5;
figure(),plot(N2O_lat,lat')
N2O_lat = [N2O_lat,lat'];
title('grassland');

%cropland
N2O_cropland_m = mean(N2O_cropland_100,2,"omitnan");
N2O_cropland_m= reshape(N2O_cropland_m,[360,720]); 
N2O_cropland_90 = imresize(N2O_cropland_m,[180,720],'nearest');
N2O_lat = mean(N2O_cropland_90,2,"omitnan"); 
lat = 89.5:-1:-89.5;
figure(),plot(N2O_lat,lat')
N2O_lat = [N2O_lat,lat'];
title('croland');

%% 3.数据分布情况
clc,clear all
load N2O_database.mat % 载入训练数据集(6016)
N2O(N2O == 0) = nan ;
N2O_Y = N2O(:,16);
N2O_Y_forest = N2O_Y(LandID == 3,:);
N2O_Y_grassland = N2O_Y(LandID == 4,:);
N2O_Y_cropland = N2O_Y(LandID == 1,:);

load meanvalue.mat
N2O_100(N2O_100 == 0) = nan;
N2O_forest_100(N2O_forest_100 == 0) = nan;
N2O_grassland_100(N2O_grassland_100 == 0) = nan;
N2O_cropland_100(N2O_cropland_100 == 0) = nan;

figure()
h1 = histogram(N2O_Y);
hold on 
h2 = histogram(N2O_100);
h1.Normalization = 'probability';
h1.BinWidth = 0.1;
h2.Normalization = 'probability';
h2.BinWidth = 0.2;
hold off

% 分组频率直方图
subplot(2,2,1)
h3 = histogram(log(exp(N2O_forest_100)*365*0.000001));
h3.Normalization = 'pdf';
h3.BinWidth = 0.1;
axis([-5 -1 0 0.13]);
text(-4.8,0.12,'forest','FontSize',12)
xlabel('ln(N_2O, g m^-^2 year^-^1)')

subplot(2,2,2)
h5 = histogram(log(exp(N2O_cropland_100)*365*0.000001));
h5.Normalization = 'pdf';
h5.BinWidth = 0.1;
axis([-5 -1 0 0.03]);
text(-4.8,0.028,'cropland','FontSize',12)
xlabel('ln(N_2O, g m^-^2 year^-^1)')

subplot(2,2,3)
h4 = histogram(log(exp(N2O_grassland_100)*365*0.000001));
h4.Normalization = 'pdf';
h4.BinWidth = 0.1;
axis([-5 -1 0 0.05]);
text(-4.8,0.045,'grassland','FontSize',12)
xlabel('ln(N_2O, g m^-^2 year^-^1)')

subplot(2,2,4)
h4 = histogram(log(exp(N2O_100)*365*0.000001));
h4.Normalization = 'pdf';
h4.BinWidth = 0.1;
axis([-5 -1 0 0.18]);
text(-4.8,0.17,'global','FontSize',12)
xlabel('ln(N_2O, g m^-^2 year^-^1)')

print(gcf,'Histogram','-dpng','-r600')


%% Spearman rank corelations between predictor varibles and N2O emission across ecosystems
clc,clear
load X_predict.mat

% land cover data 2020
Input_way1 ='D:\研究生学习\氮循环\N N2O\data\Modis land cover-2016\MCD12C1.A2020001.061.2022172062638.hdf';    
		%hdf文件在电脑中的保存位置
Input_way2 ='/MOD12C1/Data Fields/Majority_Land_Cover_Type_1'; 
		%第一种分类数据在hdf文件中的位置         
Landcover2 = hdfread(Input_way1,Input_way2, 'Index', {[1  1],[1  1],[3600  7200]});
Landcover2 = imresize(Landcover2,[360,720],'nearest');%统一图像尺寸,采用最近邻插值算法
Landcover2 = reshape(Landcover2,259200,1);

load meanvalue.mat
N2O_100(N2O_100 == 0) = nan;
N2O_global_mean1 = mean(N2O_100,2,"omitnan");

Spear_X_predict = X_predict(Landcover2 == 1 | Landcover2 == 2 | Landcover2 == 3 | Landcover2 == 4 | Landcover2 == 5 ...
     | Landcover2 == 8 | Landcover2 == 9 | Landcover2 == 10 | Landcover2 == 12 | Landcover2 == 13 | Landcover2 == 14,:);
Spear_N2O_Y = N2O_global_mean1(Landcover2 == 1 | Landcover2 == 2 | Landcover2 == 3 | Landcover2 == 4 | Landcover2 == 5 ...
     | Landcover2 == 8 | Landcover2 == 9 | Landcover2 == 10 | Landcover2 == 12 | Landcover2 == 13 | Landcover2 == 14);

spearman = [Spear_X_predict,Spear_N2O_Y];
spearman(find(isinf(spearman)))=NaN;
spearman = fillmissing(spearman,'movmean',41673);%取均值替代缺失值
spearman = im2double(spearman);

CMP = corrMatPlot(spearman,'Format','tril','Type','sq');
CMP=CMP.setLabelStr({'MAT','MAP','BD','pH','N2O','Nfer'});
CMP = CMP.draw();

spearman = spearman(:,1:14);
spearman_forest = spearman(LandID == 3,:);
spearman_grassland = spearman(LandID == 4,:);
spearman_cropland = spearman(LandID == 1,:);

%% 预测变量在文献数据和在数据产品的分布差异比较
clc,clear all
load N2O_database_补插值-包括NO3NH4.mat  %插补后数据,extract
N2O_e = N2O;
load N2O_database_origin.mat   %原始文献数据,origin
N2O_o = N2O;
Nfer_nan = N2O_e(:,13); %原始数据里没有Nfer，这里用提取的来绘图，在说明时需要告知原始数据没有
N2O_o = N2O_o(:,[1:12]);
N2O_o = [N2O_o,Nfer_nan];

%load X_predict.mat
% N2O_X_compare =[N2O_e(:,[1:12]);N2O_o(:,[1:12])];
% xlswrite('N2O_predictor_compare',N2O_X_compare);

% 一些基础设置
scatterSep='off'; % 是否分开绘制竖线散点
totalRatio='on';  % 是否各组按比例绘制
% 配色列表
colorList=[48 176 124;43 112 145]./255;

name = ["MAT,℃",'MAP,mm','BD','pH','SOC,ln(g*kg-1)','TN,ln(g*kg-1)','TP,ln(mg*kg-1)','SM,v/v%','MBC,ln(mg*kg-1)','MBN,ln(mg*kg-1)','NO3-,ln(mg*kg-1)','NH4+,ln(mg*kg-1)','Nfer, g*m-2*year-1'];
group = ["Extracted","Original"]

for i = 1:13
subplot(3,5,i)

Data(1).X= N2O_e(:,i);
Data(2).X= N2O_o(:,i); 

% 图像绘制
ax=gca;hold on
N=length(Data);
areaHdl(N)=nan;
lgdStrs{N}='';

% 计算各类数据量
K=arrayfun(@(x) length(x.X),Data);
% 循环绘图
for n=1:N
    [f,xi]=ksdensity(Data(n).X);
    if strcmp(totalRatio,'on')
        f=f.*K(n)./sum(K);
    end
    areaHdl(n)=area(xi,f,'FaceColor',colorList(n,:),...
        'EdgeColor',colorList(n,:),'FaceAlpha',.5,'LineWidth',1.5);
    lgdStrs{n}=[group(n)];
end

% 绘制图例
lgd=legend(areaHdl,lgdStrs{:});
lgd.AutoUpdate='off';
lgd.Location='best';

% 坐标区域修饰
ax.Box='on';
ax.BoxStyle='full';
ax.LineWidth=1;
ax.FontSize=11;
ax.FontName='Arial';
ax.TickDir='out';
ax.TickLength=[.005,.1];
ax.YTick(ax.YTick<-eps)=[];
% ax.Title.String= 'title';
ax.Title.FontSize=14;
ax.XLabel.String=name(i);
ax.YLabel.String='Probability density';

% 绘制基准线及框线
fplot(@(t)t.*0,'Color',ax.XColor,'LineWidth',ax.LineWidth);

end

print(gcf,'Predictor_probability density','-dpng','-r600')
