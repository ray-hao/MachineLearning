data = load('dataset.txt');

X = data(:, [1 2]);
y = data(:, 3);

accepted = find(y == 1);
rejected = find(y == 0);

plot(X(accepted, 1), X(accepted, 2), 'bo');
hold on;
plot(X(rejected, 1), X(rejected, 2), 'rx');
xlabel('Exam 1 score');
ylabel('Exam 2 score');
title('Exam scores vs. Acceptance Decision');
legend('= accepted', '= rejected');
hold off;

% [m, n] = size(X);

X = [ones(length(y), 1) X];
theta = zeros(3, 1);

costFunction(X, y, theta);

theta = gradientDescent(X, y, theta, 0.001, 500000);
% 500000 iterations! Gradient Descent was not very efficient for this
% dataset. Any greater or less alpha caused divergence of cost function and
% any  possible additional changes were negligible

plotDesicionBoundary(theta, X, y);
hold on;
xlabel('Exam 1 score')
ylabel('Exam 2 score')
legend('Admitted', 'Not admitted')
hold off;
