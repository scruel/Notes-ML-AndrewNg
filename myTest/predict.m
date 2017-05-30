function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

h1 = sigmoid([ones(m, 1) X] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');
[dummy, p_temp] = max(h2, [], 2);
%将结果进行了再效验，去除了绝对不可能的情况
	for i = 1 : m
		if h2(i,p_temp(i)) < 0.5 / num_labels
			p_temp(i) = -1;
		end
	end
p = p_temp;
% =========================================================================


end
