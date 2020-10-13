function cost = costFunction(X, y, theta)

m = length(y);
predictions = sigmoid(X * theta);

cost = (1/m) * sum((-y.*(log(predictions))) - ((1 - y).*log(1 - predictions)));

end
