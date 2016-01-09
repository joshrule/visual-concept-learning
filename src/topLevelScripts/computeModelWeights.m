function y = computeModelWeights(ms)
    y = nan(length(ms),size(ms{1}.SVs,2)+1);
    parfor i = 1:length(ms)
        w = ms{i}.SVs' * ms{i}.sv_coef;
        b = -ms{i}.rho;

        if ms{i}.Label(1) == 0
            y(i,:) = [-w; -b];
        else
            y(i,:) = [w; b];
        end
    end
end
