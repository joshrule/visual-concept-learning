function results = binary_log_regression(X_tr,y_tr,w_tr,X_te,y_te,options)
% ys should be vectors

    % save the data for classification
    dir = ensureDir([options.dir '/']);
    datafile = [dir 'data.mat'];
    save(datafile,'y_tr','y_te','X_tr','X_te','w_tr');

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
