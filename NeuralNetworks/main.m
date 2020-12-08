load('ex4data1.mat')
m = size(X, 1);

sel = randperm(size(X, 1));
sel = sel(1:100);
displayData(X(sel, :));
hold;

load('ex4weights.mat');

input_layer_size  = 400;  
hidden_layer_size = 25;   
num_labels = 10;         

nn_params = [Theta1(:) ; Theta2(:)];

%lambda = 0;
%J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);
%fprintf('Cost at parameters (loaded from ex4weights): %f', J);

lambda = 1;
J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);
fprintf('Cost at parameters (loaded from ex4weights): %f', J);

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

%checkNNGradients;

lambda = 3;
%checkNNGradients(lambda);

debug_J  = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);
fprintf(' Cost at (fixed) debugging parameters (w/ lambda = 3): %f', debug_J);

displayData(Theta1(:, 2:end));
