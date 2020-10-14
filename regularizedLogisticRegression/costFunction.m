function cost = costFunction(X, y, theta, lambda)

m = length(y);
prediction = sigmoid(X*theta);

cost = (1/m) * sum(-y.*log(prediction) - (1-y).*(log(1 - prediction))) + (lambda*(1/2)*(1/m)*(sum(theta(2:end, 1).^2)));

end
