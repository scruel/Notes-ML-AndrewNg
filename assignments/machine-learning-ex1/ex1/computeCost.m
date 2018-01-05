function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples
% You need to return the following variables correctly 

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

J = 1 / 2 / m * sum((X * theta - y) .^ 2);


%JVal = 1 / 2 / m * (X * theta - y)' * (X * theta -y);
%gradient = theta - 1 / m * (X' * (X * theta - y));
% =========================================================================

end
