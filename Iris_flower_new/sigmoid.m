function [ret_z] = sigmoid (z)
  ret_z = 1./(1 + exp(-z));

end
