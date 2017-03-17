function [X_out,means_out,stds_out] = standardize(X,means_in,stds_in)

    if nargin < 3
        stds = std(X,0,1);
    else
        stds = stds_in;
    end

    if nargin < 2
        means = mean(X,1);
    else
        means = means_in;
    end

    X_out = (X - repmat(means,size(X,1),1)) ./ repmat(stds,size(X,1),1);

    if nargout > 1
        means_out = means;
    end

    if nargout > 2
        stds_out = stds;
    end

end
