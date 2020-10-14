function output = createVariables(X1, X2)

output = ones(size(X1(:, 1)));
degree = 5;

for a = 1:degree
    for b = 0:a
        output(:, end+1) = (X1.^(a-b)).*(X2.^b);
    end
end

output;

end