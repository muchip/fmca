relu = @(x) exp(-x.^2);

x = [-10:0.1:10];
plot(x,relu(relu(relu(x)))-relu(x))