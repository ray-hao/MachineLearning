function [theta, costs] = gradientDescent(X, y, theta, alpha, iterations)

m = length(y);
costs = zeros(iterations, 1);

for i = 1:iterations
    theta = theta - (alpha) * (1/m) * ((X') * (sigmoid(X*theta) - y));
    costs(i) = costFunction(X, y, theta);
end

costs(iterations) %check the cost of the final iteration

end