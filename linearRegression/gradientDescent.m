function [theta, costs] = gradientDescent(X, y, theta, alpha, iterations)

m = length(X);
costs = zeros(iterations, 1);

for i = 1:iterations
    theta = theta - (alpha) * (1/m) * ((X') * ((X) * (theta) - y));
    costs(i) = costFunction(X, y, theta);
end

costs;

end
