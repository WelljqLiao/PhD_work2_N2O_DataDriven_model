% -*- coding: GBK -*-
% Created on Nov 1 2022 by Jiaqiang Liao
clc,clear all

% 探究2000→2020年土地利用变化导致的N2O排放变化
% 利用MODIS land cover type由cropland转变为forest和grassland的区域的思路
% 关键在于得到结果的量有多大，有多特殊
%% 1.读取2020和2001土地利用数据

% land cover data 2001
Input_way1 ='D:\研究生学习\氮循环\N N2O\data\Modis land cover-2016\MCD12C1.A2001001.061.2022146170409.hdf';    
		%hdf文件在电脑中的保存位置
Input_way2 ='/MOD12C1/Data Fields/Majority_Land_Cover_Type_1'; 
		%第一种分类数据在hdf文件中的位置         
Landcover_2001 = hdfread(Input_way1,Input_way2, 'Index', {[1  1],[1  1],[3600  7200]});
Landcover_2001 = imresize(Landcover_2001,[360,720],'nearest');%统一图像尺寸,采用最近邻插值算法
Landcover_2001 = reshape(Landcover_2001,259200,1);

% land cover data 2020
Input_way1 ='D:\研究生学习\氮循环\N N2O\data\Modis land cover-2016\MCD12C1.A2020001.061.2022172062638.hdf';    
		%hdf文件在电脑中的保存位置
Input_way2 ='/MOD12C1/Data Fields/Majority_Land_Cover_Type_1'; 
		%第一种分类数据在hdf文件中的位置         
Landcover_2020 = hdfread(Input_way1,Input_way2, 'Index', {[1  1],[1  1],[3600  7200]});
Landcover_2020 = imresize(Landcover_2020,[360,720],'nearest');%统一图像尺寸,采用最近邻插值算法
Landcover_2020 = reshape(Landcover_2020,259200,1);

%% 2.计算土地里利用转移矩阵
%设natural ecosystems为A，包括了forest和grassland，分类号有1~5，8~10
%设croplands为B，分类号有12~14

%找出两个时间段的分类信息
A_2001 = find(Landcover_2001 ==1 | Landcover_2001 == 2 | Landcover_2001 == 3 | Landcover_2001 == 4 ...
    | Landcover_2001 ==5 | Landcover_2001 ==8 | Landcover_2001 == 9 | Landcover_2001 == 10);
B_2001 = find(Landcover_2001 >= 12 & Landcover_2001 <= 14);
A_2020 = find(Landcover_2020 ==1 | Landcover_2020 == 2 | Landcover_2020 == 3 | Landcover_2020 == 4 ...
    | Landcover_2020 ==5 | Landcover_2020 ==8 | Landcover_2020 == 9 | Landcover_2020 == 10);
B_2020 = find(Landcover_2020 >= 12 & Landcover_2020 <= 14);

AtoB = intersect(A_2001,B_2020);    %取交集，找出由自然生态系统变为农田的区域
BtoA = intersect(B_2001,A_2020);    %取交集，找出由农田变为自然生态系统的区域

Land_AtoB = single(nan*[1:259200]');
Land_BtoA = single(nan*[1:259200]');

Land_AtoB(AtoB) = N2O_predict_croplands(AtoB);
Land_BtoA(BtoA) = N2O_predict_natural(BtoA);
Land_AtoB = reshape(Land_AtoB,[360,720]);
Land_BtoA = reshape(Land_BtoA,[360,720]);

[lon_2001,lat_2001] =  find(Land_AtoB>0);

weights = Land_AtoB;

geodensityplot(lat_2001,lon_2001,weights,'FaceColor','interp')

N2O_blank = single(nan*[1:259200]');
N2O_blank(Landcover >= 6 & Landcover <= 7) = 0;
N2O_blank(Landcover == 11) = 0;
N2O_blank(Landcover >=16) = 0;
N2O_blank = reshape(N2O_blank,[360,720]);

figure1 = figure
axes1 = axes('Parent',figure1);
hold(axes1,'on');
surf(N2O_blank','EdgeColor','#BEBEBE');
mesh(Land_AtoB','Parent',axes1);
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