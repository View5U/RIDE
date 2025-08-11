function noisy_target = generate_idn_noise(target, Outputs, numn)
if size(target, 1) ~= size(Outputs, 1) || size(target, 2) ~= size(Outputs, 2)
    error('The size of the target and Outputs matrices must be the same.');
end

[n, q] = size(target);
noisy_target = target;

num_ones = sum(target(:) == 1);
num_zeros = sum(target(:) == 0);
r = num_ones / num_zeros;

num_false_positive = round(numn * r / (1 + r));
num_false_negative = numn - num_false_positive;

max_fp_per_row = sum(target == 0, 2);
max_fn_per_row = sum(target == 1, 2);

np = zeros(n, 1);
nn = zeros(n, 1);

available_rows_fp = find(max_fp_per_row > 0);
while num_false_positive > 0 && ~isempty(available_rows_fp)
    row = available_rows_fp(randi(length(available_rows_fp)));
    if np(row) < max_fp_per_row(row)
        np(row) = np(row) + 1;
        num_false_positive = num_false_positive - 1;
    end
    available_rows_fp = find(max_fp_per_row - np > 0);
end

available_rows_fn = find(max_fn_per_row > 0);
while num_false_negative > 0 && ~isempty(available_rows_fn)
    row = available_rows_fn(randi(length(available_rows_fn)));
    if nn(row) < max_fn_per_row(row)
        nn(row) = nn(row) + 1;
        num_false_negative = num_false_negative - 1;
    end
    available_rows_fn = find(max_fn_per_row - nn > 0);
end

for i = 1:n
    if np(i) > 0
        probs = custom_softmax(Outputs(i, :).*(~target(i, :)));
        for j = 1:np(i)
            while sum(~noisy_target(i,:)) ~= 0
                idx = randsample(q, 1, true, probs);
                if noisy_target(i, idx) == 0
                    noisy_target(i, idx) = 1;
                    break;
                end
            end
        end
    end
end

for i = 1:n
    if nn(i) > 0
        probs = custom_softmax(((ones(size(Outputs(i, :)))-Outputs(i, :))).*(target(i, :)));
        for j = 1:nn(i)
            while sum(noisy_target(i,:)) ~= 0
                idx = randsample(q, 1, true, probs);
                if noisy_target(i, idx) == 1
                    noisy_target(i, idx) = 0;
                    break;
                end
            end
        end
    end
end
end

function s = custom_softmax(x)
non_zero_indices = x ~= 0;
x_non_zero = x(non_zero_indices);
exp_x_non_zero = exp(x_non_zero - max(x_non_zero));
softmax_values = exp_x_non_zero / sum(exp_x_non_zero);
s = zeros(size(x));
s(non_zero_indices) = softmax_values;
end
