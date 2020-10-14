function [theta, costs] = gradientDescent(X, y, theta, alpha, lambda, iterations)

m = length(y);
costs = zeros(iterations, 1);

for i = 1:iterations
   theta = theta - (alpha)*(1/m)*(X'*(sigmoid(X*theta) - y)) - (alpha)*(lambda)*(1/m)*([0; theta(2:end)]);
   costs(i) = costFunction(X, y, theta, lambda);
end

costs(iterations)

end

