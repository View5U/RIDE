function Pre_Labels = getPredict(train_data, test_data, W, Y, tuneThreshold)
Outputs = (test_data*W)';
if tuneThreshold == 0
    fscore     = (train_data*W);
    [tau,  ~]  = TuneThreshold(fscore', Y', 0, 1);
    Pre_Labels = Predict(Outputs, tau);
else
    Pre_Labels = double(Outputs>tuneThreshold);
end
end
