function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

	for iter = 1:num_iters

		% ====================== YOUR CODE HERE ======================
		% Instructions: Perform a single gradient step on the parameter vector
		%               theta. 
		%
		% Hint: While debugging, it can be useful to print out the values
		%       of the cost function (computeCostMulti) and gradient here.
		%
		theta = theta - alpha / m * (X' * (X * theta - y));
		%X * theta 代表了hθ
		%X' (hθ-y)表示将每一个X中的元素(n * m)与(hθ-y)(m * 1)相乘，从而得到新的theta矩阵(n * 1)


    % ============================================================

		% Save the cost J in every iteration    
		J_history(iter) = computeCostMulti(X, y, theta);

	end

end
