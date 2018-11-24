%This is the iris flower detection project
%we are given to identify 3 types of flowers based on a training set
%The functions included are 
%     lrCostFunction.m (logistic regression cost function)
%     oneVsAll.m
%     predictOneVsAll.m
%     sigmoid.m
%     fmincg.m
clear;close all;clc;
%   loading the data from iris_flower.txt
fprintf('loading the data...\n');
data = load('iris_flower.txt');
%   some useful parameters
m = size(data,1);
n = size(data,2);
%   store all the features in X, and the result in y
X = data(:, 1:4);
y = data(:, 5);
fprintf('Program paused. Press enter to continue.\n');
pause;

X = [ones(m,1) X];
fprintf('Running oneVsall Algo...\n');
%   number of classes signifies different types of flowers 
num_of_classes = 3;
lambda = 0.1;
[all_theta] = oneVsAll(X,y,num_of_classes,lambda);

fprintf('Program paused. Press enter to continue.\n');
pause;
fprintf('==================Running prediction===================\n');
p = predictOneVsAll(X, all_theta);
accuracy = mean((p == y)*100);
fprintf('Accuracy of the algorithm = %f\n',accuracy);
fprintf('Program paused. Press enter to continue.\n');
pause;
fprintf('END....\n');






