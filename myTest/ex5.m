
clear ; close all; clc


fprintf('Loading and Visualizing Data ...\n')
input_layer_size  = 140;
hidden_layer_size = 200; 
num_labels = 18;
data = load('w2v_tr.txt');
X = data(:, 2:end);
y = data(:, 1);

data = load('w2v_val.txt');
Xval = data(:, 2:end);
yval = data(:, 1);

data = load('w2v_test.txt');
Xtest = data(:, 2:end);
ytest = data(:, 1);
m = size(X, 1);

fprintf(string"'\nTraining Neural Network... \n'")
options = optimset('MaxIter',100);
lambda = 0.02;
%checkNNGradients(lambda)
[nn_params, cost] = trainNN(input_layer_size, ...
									hidden_layer_size, ...
									num_labels, ...
									options, ...
									X, y, lambda);

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
size(cost)
plot(1:size(cost,1),cost);

ylabel('J Cost')
xlabel('Iterator')
fprintf('# Training Finished...');
pred = predict(Theta1, Theta2, X)
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
pred = predict(Theta1, Theta2, Xval)
fprintf('\nValidation Set Accuracy: %f\n', mean(double(pred == yval)) * 100);
pred = predict(Theta1, Theta2, Xtest)
fprintf('\nTest Set Accuracy: %f\n', mean(double(pred == ytest)) * 100);
pause;

%%=======================================================

[error_train, error_val] = ...
    learningCurve(input_layer_size, ...
                            hidden_layer_size, ...
                            num_labels, ...
							options, ...
							X, y, Xval, yval, lambda);
plot(1:m, error_train, 1:m, error_val);
title('Learning curve for linear regression')
legend('Train', 'Cross Validation')
xlabel('Number of training examples')
ylabel('Error')
axis([0 41 0 60])

fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
for i = 1:m
    fprintf('   %d %f %f\n', i, error_train(i), error_val(i));
end

fprintf('Program paused. Press enter to continue.\n');
pause;
%% =======================================================

% [lambda_vec, error_train, error_val] = ...
    % validationCurve(input_layer_size, ...
                            % hidden_layer_size, ...
                            % num_labels, ...
							% options, ...
							% X, y, Xval, yval);
% close all;
% plot(lambda_vec, error_train, lambda_vec, error_val);
% legend('Train', 'Cross Validation');
% xlabel('lambda');
% ylabel('Error');
% fprintf('lambda Train Error Validation Error\n');
% for i = 1:length(lambda_vec)
	% fprintf(' %f %f %f\n', ...
            % lambda_vec(i), error_train(i), error_val(i));
% end

% fprintf('Program paused. Press enter to continue.\n');
% pause;
