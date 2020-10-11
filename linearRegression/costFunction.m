function J = costFunction(X, y, theta)

m = length(X);
predictions = X * theta;
squaredErrors = (predictions - y) .^ 2;

J = (1/2) * (1/m) * sum(squaredErrors);

end
