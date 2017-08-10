function [X_out,means_out,stds_out] = standardize(X,w,means_in,stds_in)
    N = size(X,1);

    if isequal(w,[])
        w = ones(N,1);
    else
        w = reshape(w,[],1);
    end
    
    sumw = sum(w);


    if nargin < 3
        means = (w'*X)./sumw;
    else
        means = means_in;
    end

    if nargin < 4
        stds = sqrt((w'*(X-repmat(means,N,1)).^2)./(sumw-1));
    else
        stds = stds_in;
    end

    X_out = (X - repmat(means,N,1)) ./ repmat(stds,N,1);

    if nargout > 1
        means_out = means;
    end

    if nargout > 2
        stds_out = stds;
    end
end
