function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
m = size(X, 1);
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
temp_y = zeros(m,num_labels);
for i = 1 : m
	temp_y(i,y(i)) = 1;
end
p = zeros(size(X, 1), 1);
X = [ones(m, 1), X];
a2 = sigmoid(X * Theta1');
a2 = [ones(m, 1), a2];
hx = sigmoid(a2 * Theta2');
%无需对hx的结果进行统一化（提取max），因为hx对每一个值的预估都是有用的数据，可以用来计算J
J = 1 / m * sum(sum(-temp_y .* log(hx) - (1 - temp_y) .* log(1 - hx))) + lambda / (2 * m) * (sum((Theta1(:, 2:end) .^ 2)(:)) + (sum((Theta2(:, 2:end) .^ 2)(:))));
delta_3 = hx - temp_y;
delta_2 = delta_3 * Theta2.* a2 .* (1 - a2);
delta_2 = delta_2(:, 2:end);
Theta2_grad = 1 / m * (Theta2_grad + delta_3' * a2);
Theta1_grad = 1 / m * (Theta1_grad + delta_2' * X);
Theta2_grad(:, 2:end)  = Theta2_grad(:, 2:end) + lambda / m * (Theta2(:, 2:end));
Theta1_grad(:, 2:end)  = Theta1_grad(:, 2:end) + lambda / m * (Theta1(:, 2:end));
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
