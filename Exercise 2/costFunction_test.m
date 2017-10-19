%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

clear all;
clc;

theta = [0;0;0]
X = [1,2,3; 1,4,5; 1,3,6];
y = [1;0;0]
% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));



% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
h = sigmoid(X*theta);
J = 1/m * sum(-1*y .* log(h) - (1-y) .* log(1 - h));

g = 1;

while g <= size(theta)
    grad(g) = 1/m * sum((h - y) .* X(:, g));
    g = g + 1;
    
    if (g == 2)
        disp("Sum is");
        disp((h-y));
    end
end

% =============================================================

