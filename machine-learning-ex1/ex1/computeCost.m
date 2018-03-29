function J = computeCost(X, y, theta)

% COMPUTECOST Compute cost for linear regression
%  J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%  parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% sum up squares
squares_sum = 0;
for i = 1:m,

    % calculate hypothesis
    h = theta(1) * X(i,1) + theta(2) * X(i,2);

    % sum square
    squares_sum = squares_sum + (h - y(i,1)) ^ 2;

end;

% finally calculate the cost
J = (1 / (2 * m)) * squares_sum;

% =========================================================================

end
