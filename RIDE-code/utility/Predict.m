function predict_target = Predict(Outputs, tau)
predict_target = zeros(size(Outputs));
[num_class, num_ins] = size(Outputs);
for c = 1:num_class
    predict_target(c,:) = Outputs(c,:) >= tau(1,c);
end

predict_max = Outputs;

for ins=1:num_ins
    for c = 1:num_class
        if( predict_max(c,ins) < max(predict_max(:,ins)) )
            predict_max(c,ins) = 0;
        end
    end
end

predict_max = sign(predict_max);

for ins=1:num_ins
    if( sum(predict_target(:,ins))==0 )
        predict_target(:,ins) = predict_max(:,ins);
    end
end