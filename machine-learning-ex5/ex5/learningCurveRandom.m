

function [error_train_rand, error_val_rand] = learningCurveRandom (X_poly, y, X_poly_val, yval, lambda)
  m1 = size(X_poly, 1);
  m2 = size(X_poly_val, 1);
  for i = 1:m1
    % error for the i chosen examples
    error_dummy_train = 0;
    error_dummy_val = 0;
    for  j = 1:50
    rand_ind_X = randperm(m1);
    rand_ind_Xv = randperm(m2);
    
    % randomly selected i examples each
    % from X_poly and X_val
    sel_X = X_poly(rand_ind_X(1:i), :);
    sel_Xv = X_poly_val(rand_ind_Xv(1:i), :);
    
    % y values for the selected indices of train
    y_X = y(rand_ind_X(1:i));
    % y values for the selected indices of validation
    y_Xv = yval(rand_ind_Xv(1:i));
    
      theta = trainLinearReg(sel_X, y_X, lambda);
      error_dummy_train += linearRegCostFunction(sel_X, y_X, theta, 0);
      error_dummy_val += linearRegCostFunction(sel_Xv, y_Xv, theta, 0);
      end
      error_train_rand(i) = error_dummy_train/50;
      error_val_rand(i) = error_dummy_val/50;
       end
      
      
      

end
