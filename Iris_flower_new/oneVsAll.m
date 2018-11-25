% oneVsAll algorithm divides the dataset into
% 2 classes and finds optimum theta
% values for the respective classes

function [all_theta] = oneVsAll (X, y, num_of_classes, lambda)
  m = size(X, 1);
  n = size(X, 2);
  all_theta = zeros(num_of_classes, n);
  
  
  for i = 1: num_of_classes
    initial_theta = zeros(n, 1);
    options = optimset('GradObj','on','MaxIter',50);
    % (y == i)divides the training set into 2 distinct values
    % to get the optimum parameters for theta
    [theta] = fmincg(@(t)(lrCostFunction(t,X,y == i,lambda)),initial_theta,options);
    all_theta(i,:) = theta;
    
    end
  

end
