clear;

%% Settings

data_name = 'dataset_name';

% cal500
% corel16k001
% mediamill
% medical
% scene
% slashdot
% yeast

noiseRate = 0.5;

lambda1 = 0.1;
lambda2 = 0.1;
eta     = 10;

rho = 1e0;     % ADMM augmentation parameter
lr = 1e-3;     % max learning rate of updating W
tol = 1e-3;    % min loss margin
maxIter = 100;

% Select a method of data preprocessing, from: zscore, kernel, minmax, none
preprocessing = "minmax";

% Select a method of classifier induction, from: knn, forward
classifier = "knn";

bias = 0.01; % bias (-0.5, +0.5) must be carefully tuned if knn is selected

%% Load data

load(['dataset\' data_name '_noisy.mat']);
target(target==-1)=0;
N = length(target);
GLs = sum(sum(target))/N;

%% Data preprocessing

if preprocessing == "zscore"
    data = zscore(data);
elseif preprocessing == "kernel"
    ker  = 'rbf';
    par = 1*mean(pdist(data));
    X = kernelmatrix(ker,data',data',par);
    data = X;
elseif preprocessing == "minmax"
    [data, settings] = mapminmax(data');
    data = data';
elseif preprocessing == "none"
end

%% Generalization of IDN

if noiseRate ~= 0
    numn = GLs*noiseRate*N;
    Predicts = generate_idn_noise(target, Outputs, numn);
end

%% K-fold experiments

fold = 5;
indices = crossvalind('Kfold',1:N,fold);
for id=1:N
    indices(id)=mod(id, fold)+1;
end

Metrics = zeros(6,fold);
tic;
for f = 1:fold
    test_idxs = (indices==f);
    test_data=data(test_idxs,:);
    test_target=target(test_idxs,:);
    noisy_test_target=Predicts(test_idxs,:);
    train_idxs = ~test_idxs;
    train_data=data(train_idxs,:);
    train_target=target(train_idxs,:);
    noisy_train_target=Predicts(train_idxs,:);
    train_data(isnan(train_data))=0;
    test_data(isnan(test_data))=0;

    fprintf("Running " + "CrossValidation: " + f + "\n");  alg = 'RIDE';
    model = solver_RIDE(train_data, noisy_train_target, lambda1, lambda2, eta, rho, maxIter, lr, tol); W = model.W; V=model.V;

    test_data = [test_data, ones(size(test_target,1),1)]; %#ok<AGROW>
    train_data = [train_data, ones(size(train_data,1),1)]; %#ok<AGROW>


    if classifier == "knn"
 
        Y = noisy_train_target;
        T = Y;
        k_knn = min(500, size(test_data,1));
        q = size(Y, 2);
        label_weights = q * sum(Y, 1) / sum(Y(:));
        Wproj      = model.W';
        train_proj = Wproj * train_data';
        n_test     = size(test_data, 1);
        Scores     = zeros(n_test, q);
        PredLabels = false(n_test, q);
        label_mask = any(Y, 1);
        for i = 1:n_test
            test_vec = Wproj * test_data(i, :)';
            diffs = train_proj - test_vec;
            dists = label_weights * (diffs .^ 2);
            [sorted_dists, idx] = sort(dists);
            valid_idx = find(sorted_dists > 0, k_knn, 'first');
            nn_idx    = idx(valid_idx);
            weights   = 1 ./ sorted_dists(valid_idx);
            T_sub     = T(nn_idx, label_mask);
            score_vec = (weights * T_sub) / sum(weights);
            Scores(i, label_mask) = score_vec;
            thresh = min(score_vec) + 0.5*(max(score_vec)-min(score_vec)) + bias;
            PredLabels(i, label_mask) = score_vec >= thresh;
        end
        Outputs    = Scores';
        Pre_Labels = PredLabels';

    elseif classifier == "forward"
        Outputs = (test_data*W)';
        Pre_Labels = getPredict(train_data, test_data, W, train_target, 0);
    else
        fprintf("Must be choosen from knn or forward");
    end

    Result_MLC = EvaluationAll(Pre_Labels, Outputs, test_target');
    HammingLoss = Result_MLC(1, 1);
    MicroF1 = Result_MLC(11, 1);
    AveragePrecision = Result_MLC(12, 1);
    OneError = Result_MLC(13, 1);
    RankingLoss = Result_MLC(14, 1);
    Coverage = Result_MLC(15, 1);

    % Evaluate
    Metrics(1,f) = HammingLoss;
    Metrics(2,f) = MicroF1;
    Metrics(3,f) = AveragePrecision;
    Metrics(4,f) = OneError;
    Metrics(5,f) = RankingLoss;
    Metrics(6,f) = Coverage;
end

Metrics = Metrics';
HammingLoss=Metrics(:,1);
MicroF1=Metrics(:,2);
AveragePrecision=Metrics(:,3);
OneError=Metrics(:,4);
RankingLoss=Metrics(:,5);
Coverage=Metrics(:,6);
fprintf("\n"+ data_name +"\n" + alg +"\n");
fprintf(' HammingLoss: %f std: %f\n MicroF1: %f std: %f\n AveragePrecision: %f std: %f\n OneError: %f std: %f\n RankingLoss: %f std: %f\n Coverage: %f std: %f\n ',mean(HammingLoss),std(HammingLoss),mean(MicroF1),std(MicroF1),mean(AveragePrecision),std(AveragePrecision),mean(OneError),std(OneError),mean(RankingLoss),std(RankingLoss),mean(Coverage),std(Coverage));
toc;

