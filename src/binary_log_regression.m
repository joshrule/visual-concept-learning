function results = binary_log_regression(dir)
% ys should be vectors

    % call the classifier
    if ~exist([dir 'results.mat'],'file')
        system(['LD_LIBRARY_PATH=/usr/lib/:/usr/local/lib/:/usr/local/cuda/lib:/usr/local/cuda/lib64 ' ...
        'python binarylr.py ' dir ' &> ' dir 'log.log']);
    end

    results = load([dir 'results.mat']);

end

function [X,y,order] = shuffle(X,y)
    order = randperm(length(y));
    X = X(order,:);
    y = y(order); % subtract one for Python's 0-indexing
end
