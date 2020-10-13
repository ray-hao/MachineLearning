function a = sigmoid(b)

e = 2.718;
a = (1 + e.^(-1 .* b)).^-1;

end
