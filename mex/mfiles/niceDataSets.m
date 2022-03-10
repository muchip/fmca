clear all;
close all;
N = 1000

%theta = np.sqrt(np.random.rand(N))*2*pi # np.linspace(0,2*pi,100)
theta = sqrt(rand(N,1)) * 2 * pi;
r_a = 2 * theta + pi;
%data_a = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T
data_a = [cos(theta) .* r_a, sin(theta) .* r_a];
%x_a = data_a + np.random.randn(N,2)
x_a = data_a + 0.75 * randn(N,2);
plot(x_a(:,1),x_a(:,2),'ro')

r_b = -2*theta - pi;

% data_b = np.array([np.cos(theta)*r_b, np.sin(theta)*r_b]).T
data_b = [cos(theta) .* r_b, sin(theta) .* r_b];
x_b = data_b + 0.8 * randn(N,2);

hold on;
plot(x_b(:,1),x_b(:,2),'bo')
axis square
axis tight
% x_b = data_b + np.random.randn(N,2)
% 
% res_a = np.append(x_a, np.zeros((N,1)), axis=1)
% res_b = np.append(x_b, np.ones((N,1)), axis=1)
% 
% res = np.append(res_a, res_b, axis=0)
% np.random.shuffle(res)