data = load('dataset.txt');
X = data(:, [1 2]);
y = data(:, 3);

pos = find(y==1);
neg = find(y==0);

%plot(X(pos, 1), X(pos, 2), 'bo')
%hold on;
%plot(X(neg, 1), X(neg, 2), 'rx')
%xlabel('Test 1');
%ylabel('Test 2');
%legend('Passed', 'Failed');
%title('Test Results vs. Performance');

X = createVariables(X(:, 1), X(:, 2));

[m, n] = size(X);

theta = zeros(n, 1);
lambda = 1;

theta = gradientDescent(X, y, theta, 0.5, lambda, 5000);

plotDecisionBoundary(theta, X, y)