data = load('dataset.txt');

X = data(:, 1);
X = [ones(length(X),1), X];
y = data(:,2);
theta = zeros(2,1);

costFunction(X, y, theta);
theta = gradientDescent(X, y, theta, 0.01, 2000);

plotData(X(:,2), y)
hold on
plot(X(:,2), X*theta, 'b')
legend('training data', 'linear regression')