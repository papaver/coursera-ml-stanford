function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

% NNCOSTFUNCTION Implements the neural network cost function for a two layer
% neural network which performs classification
%  [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%  X, y, lambda) computes the cost and gradient of the neural network. The
%  parameters for the neural network are "unrolled" into the vector
%  nn_params and need to be converted back into the weight matrices.
%
%  The returned parameter grad should be a "unrolled" vector of the
%  partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
K = num_labels;

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% hypothesis
a1 = [ones(m, 1) X];
z2 = a1 * Theta1';
a2 = [ones(m, 1) sigmoid(z2)];
z3 = a2 * Theta2';
a3 = sigmoid(z3);
h  = a3;

% vectorize y (expand y index into a vector (like a mask), ex: 4 = [0 0 0 1]
%  sub2ind allows selecting from indecies from a vector, so use range 1:len(y)
%  for the rows and for each row use y' to represent the column to select
Y = zeros(m, K);
Y(sub2ind(size(Y), 1:length(y), y')) = 1;

% cost :
%  use Y as a mask to pick the selection which should be one
%  then add up all the choices, and average over m, thats the cost (average error)
J_reg = lambda / (2 * m) * (sum(sum(Theta1(:, 2:end) .^ 2)) + sum(sum(Theta2(:, 2:end) .^ 2)));
J     = (1 / m) * sum(sum((log(h) .* -Y) - (log(1 - h) .* (1 - Y)))) + J_reg;

% backward propogation
d3 = a3 - Y;                                             % 2) initial error
d2 = (Theta2(:, 2:end)' * d3')' .* sigmoidGradient(z2);  % 3) hidden layer errors
Theta1_grad = (a1' * d2)' / m;                           % 4) accumulate thetas
Theta2_grad = (a2' * d3)' / m;

% add regularization
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + (lambda / m * Theta1(:, 2:end));
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + (lambda / m * Theta2(:, 2:end));

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
