% here we are going to use all the training examples and find their respective 
% classes by using the hypothesis function sigmoid
% The value in which hypothesis returns the max value is our predicted 
% value which we store in p
% Note: p is a column vector

function [p] = predictOneVsAll (X_new,all_theta)
  m = size(X_new, 1);
  p = zeros(m,1);
  h = sigmoid(X_new*all_theta');
  [max_val,max_ind] = max(h,[],2);
  p = max_ind;

end
