function model = solver_RIDE(X, Y, lambda1, lambda2, eta, rho, maxIter, lr, tol)

% Y = init(X, Y);  % optional initialization

[n,q] = size(Y);
X = [X, ones(n,1)];
[~,d] = size(X);

rhoI = 100;
distu = 0;
smooth = 0.01;
b = 1;

W = (X'*X + rhoI*eye(d))\(X'*Y) + distu*randn(d, q);
W0 = W;
V = zeros(d, q) + distu*randn(d, q);
M = zeros(d, q);
N = distu*randn(n, q);
T = Y;
A = zeros(size(Y));

distribution = X*W;
for i = 1:n
    distribution(i,:) = distribution(i,:) - min(distribution(i,:));
    distribution(i,:) = distribution(i,:)/sum(distribution(i,:));
end
D = distribution;
iter = 1;
lambdaw = size(X,1)/size(X,2);
obj = zeros(100);
while(iter<=10)
    eps = 1e-7;
    Diag = diag(1./max(sqrt(sum((W).*(W),2)),eps));
    W = (X'*X+lambdaw*Diag+rhoI*eye(d))\(X'*D);
    obj(iter) = 0.5*trace((X*W-D)'*(X*W-D))+lambdaw*sum(sqrt(sum((W).*(W),2)));
    if iter>=2 && abs(obj(iter)-obj(iter-1))<=1e-3
        break;
    end
    iter=1+iter;
end
W = 0.5*(W+W0);

eps_primal = tol;
eps_dual = tol;

s = cals(X, W, Y, smooth);
s = n * s ./ sum(s);

for k = 1:maxIter
    if mod(k, 100)==0
        if lr > 1e-7
            lr = 0.1*lr;
        else
            lr = 1e-7;
        end
    end

    % update T
    M_T = (X*W + rho*(Y - N + A/rho)) / (1 + rho);
    T = SVT(M_T, lambda2/(1+rho));
    T = max(0, T);

    % update N
    M_N = (X*V + rho*(Y - T + A/rho)) / (1 + rho);
    N = sign(M_N) .* max(abs(M_N) - lambda1/(1+rho)*s(:), 0);
    N = max(min(N, 1), -1);

    % update W
    W_prev = W;
    W = W_prev + Armijo_W(X, W_prev, V, T, eta, lr);

    % update V
    [V, M, b] = FISTA(V, X, N, W, M, eta, b);
    
    % update A
    A = A + rho * (Y - T - N);
    rho = min(rho*1.1, 1e2);

    % stop criteria
    r_para = norm(Y - T - N, 'fro');
    r_dual = norm(W - W_prev, 'fro');
    if mod(k, 1)==0 || k==1
        fprintf('Iter %d: r_para = %.4e, r_dual = %.4e\n', k, r_para, r_dual);
    end
    if (r_para < eps_primal) && r_dual < eps_dual && k>=10
        fprintf('Converged at iteration %d.\n', k);
        break;
    end
end

model.W = W; model.V = V; model.Y = Y; model.T = T; model.N = N; model.A = A;
end

function T = SVT(M, tau)
[U,S,V] = svd(M, 'econ');
S_threshold = diag(soft_threshold(diag(S), tau));
T = U * S_threshold * V';
end

function X_out = soft_threshold(X_in, tau)
X_out = sign(X_in) .* max(abs(X_in) - tau, 0);
end

function [V, M, b_new] = FISTA(V_old, X, N, W, M, eta, b_old)
eta = eta*0.1;
L = norm(X, 2)^2 + 2*eta*(norm(W, 2)^2);
eta = 1 / (L+1);
grad = X'*(X*M - N) + 2*eta * (W*(W'*M));
V_s = M - eta * grad;
V_new = soft_threshold(V_s, eta * eta);
b_new = (1 + sqrt(1 + 4*b_old^2))/2;
M = V_new + ((b_old - 1)/b_new) * (V_new - V_old);
V = V_new;
end

function dW = Armijo_W(X, W, V, T, eta, lr)
alpha_0 = lr;   sigma = 0.1;    beta = 0.5;
B = W21(W);
oldobj = 0.5*(norm(X*W - T, 'fro')^2) + eta*( trace(W'*B*W) + 0.5*norm(W'*V, 'fro')^2 );
Grad_W = X'*(X*W - T) + eta*( 2*B*W + V*(V'*W) );
p = -Grad_W;
alpha = alpha_0;
while true
    Wnew = W + alpha*p;
    Bnew = W21(Wnew);
    newobj = 0.5*(norm(X*Wnew - T, 'fro')^2) + eta*( trace(Wnew'*Bnew*Wnew) + 0.5*norm(Wnew'*V, 'fro')^2 );
    if newobj <= oldobj + sigma * alpha * trace(Grad_W' * p)
        break;
    else
        alpha = beta * alpha;
    end
end
dW = alpha * p;
end

function B = W21(W)
eps = 1e-10;
norms = sqrt(sum(W.^2, 2));
norms(norms == 0) = inf;
B = diag(1 ./ norms);
B(isinf(B)) = eps;
end

function s = cals(X, W, Y, smooth)
n = size(X, 1);
q = size(W, 2);
k = 10;
sigma = 0.5;

XA = zeros(n,1);
for i = 1:n
    XA(i,1) = norm(X(i,:));
end
X = X./XA;
n = size(X,1);
[dis, neighbor] = pdist2(X, X, 'euclidean', 'Smallest', k+1);
dis = dis(end-k+1:end,:);
neighbor = neighbor(2:end, :);
neighbor = neighbor';
datas = zeros(n, k);
for i = 1:n
    w = dis(:,i);
    w = exp(-w.^2 / sigma^2);
    w = w ./ sum(w);
    datas(i,:) = w;
end
nei = neighbor;
wei = datas;
wei = wei ./ sum(wei, 2);
nei_labels = Y(nei, :);
nei_labels = reshape(nei_labels, [n, k, q]);
Yknn = squeeze(sum(wei .* nei_labels, 2));

% Yknn = Yknn - ( min(min(Yknn)) + 0.5*(max(max(Yknn))-min(min(Yknn))) );
% Yknn = sign(Yknn);        
% Yknn = max(Yknn, 0);

yi_norm = sqrt(sum(Y.^2, 2));
yknn_norm = sqrt(sum(Yknn.^2, 2));
dot_product = sum(Y .* Yknn, 2);
s = dot_product ./ (yi_norm .* yknn_norm);
s(isnan(s)) = 0;

s = s + smooth;
end

function Y = init(X, Y) %#ok<*DEFNU> 
n = size(X, 1);
q = size(Y, 2);
k = 10;
XA = zeros(n,1);
for i = 1:n
    XA(i,1) = norm(X(i,:));
end
X = X./XA;
n = size(X,1);
[dis, neighbor] = pdist2(X, X, 'euclidean', 'Smallest', k+1);
dis = dis(end-k+1:end,:);
neighbor = neighbor(2:end, :);
neighbor = neighbor';
datas = zeros(n, k);
for i = 1:n
    w = dis(:,i);
    w = 1./w;
    w = w ./ sum(w);
    datas(i,:) = w;
end
nei = neighbor;
wei = datas;
wei = wei ./ sum(wei, 2);
nei_labels = Y(nei, :);
nei_labels = reshape(nei_labels, [n, k, q]);
Yknn = squeeze(sum(wei .* nei_labels, 2));
zero_rows = sum(Y, 2) == 0;
if any(zero_rows)
    [~, idx] = max(Yknn(zero_rows, :), [], 2);
    Y(sub2ind(size(Y), find(zero_rows), idx)) = 1;
end
end
