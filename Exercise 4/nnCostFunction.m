function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
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

X = [ones(m, 1), X];        % dim 5000 x 401
zsup2 = Theta1 * X.';       % dim 25 * 5000

asup1_all_ex = X.';       % dim 401 x 5000
asup2_all_ex = sigmoid(zsup2);    % dim 25 x 5000

asup2_all_ex = [ones(1,m); asup2_all_ex];         % hidden layer with ones added

zsup3 = Theta2 * asup2_all_ex;
y_hat = sigmoid(zsup3);    % dim 10 x 5000 

y_exp = zeros(num_labels, m);

index = 1;

while index <= m
    y_exp(y(index),index) = 1;
    index = index + 1;
end

J = 1/m * sum(sum(-1*y_exp.*log(y_hat) - (1 - y_exp).*log(1-y_hat))) + ...
    0.5*lambda/m*(sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));

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

delta_sup3_all_ex = y_hat - y_exp;    % dim 10 x 5000
Delta_sup2 = zeros(size(Theta2));     % dim 10 x 26
Delta_sup1 = zeros(size(Theta1));     % dim 25 x 401

index = 1;
while (index <= m)
    delta_sup3 = delta_sup3_all_ex(:, index);   % dim 10 x 1
    zsup2_one_ex = zsup2(:, index);             % dim 25 x 1
    delta_sup2 = Theta2(:, 2:end).' * delta_sup3 .* sigmoidGradient(zsup2_one_ex);   % dim 25 x 1
    asup2 = asup2_all_ex(:, index);        %  dim 26 x 1
    asup1 = asup1_all_ex(:, index);
    
    Delta_sup2 = Delta_sup2 + delta_sup3 * asup2.';
    Delta_sup1 = Delta_sup1 + delta_sup2 * asup1.';
    index = index + 1;
end

Theta1_grad = 1/m * Delta_sup1;
Theta2_grad = 1/m * Delta_sup2;
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
