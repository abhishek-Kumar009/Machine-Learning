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

%   store all the features in X, and the result in y
%   Only 40 training examples from each class is used to train LR
X = [data(1:40, 1:4);data(51:90,1:4);data(101:140,1:4)] ;

y = [data(1:40, 5);data(51:90,5);data(101:140,5)];
% the new dataset to be tested
X_new = [data(41:50,1:4);data(91:100,1:4);data(141:150,1:4)];
y_new = [data(41:50, 5);data(91:100,5);data(141:150,5)];
fprintf('Program paused. Press enter to continue.\n');
pause;

X = [ones(size(X,1),1) X];
X_new = [ones(size(X_new,1),1) X_new];
fprintf('Running oneVsall Algo...\n');
%   number of classes signifies different types of flowers 
num_of_classes = 3;
lambda = 0.1;
[all_theta] = oneVsAll(X,y,num_of_classes,lambda);

fprintf('Program paused. Press enter to continue.\n');
pause;
fprintf('==================Running prediction===================\n');
pass = [ones(size(data,1),1) data(:,1:4)];
p = predictOneVsAll(pass, all_theta);
accuracy = mean((p == data(:,5))*100);
fprintf('Accuracy of the algorithm = %f\n',accuracy);
fprintf('Program paused. Press enter to continue.\n');
pause;
fprintf('END....\n');






