function [J, grad] = lrCostFunction (theta, X, y, lambda)

m = size(X, 1);
n = size(X, 2);
J = 0;
grad = zeros(size(theta));

h = sigmoid(X*theta);
J = (1/m)*(-y'*log(h) - (1 - y)'*log(1 - h)) + (lambda/(2*m))*(sum(theta.^2) - theta(1)^2);
grad = (1/m).*(X'*(h - y)) + (lambda/m).*(theta);
grad(1) = (1/m)*(sum(h - y));




end
