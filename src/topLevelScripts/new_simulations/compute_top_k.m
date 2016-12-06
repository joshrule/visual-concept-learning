function compute_top_k(outDir,imgs,labels,type,ks)
% compute_top_k(outDir,imgs,labels,type,ks)
%
% give top_k judgments
%
% Labels should be numeric and correspond to the labels used during
% training for the network.
    accFile = [outDir 'feature_validation_top_k_judgments.csv'];
    judFile = [outDir 'feature_validation_top_k_accuracies.csv'];
    if ~exist(accFile,'file') || ~exist(judFile,'file')

        imgs = row_vector(imgs);
        labels = row_vector(labels);
        ks = row_vector(ks);

        max_k = max(ks);

        judgments = nan(length(imgs),numel(ks));
        top_k     = nan(length(imgs),max_k);

        for iImg = 1:length(imgs)

            % load the correct activation
            [d,f,e] = fileparts(imgs{iImg});
            load([d '/' f '.' type '_mat'],'c2','-mat');

            % which k labels are most likely?
            [~,idxs] = sort(c2,'descend');
            top_k(iImg,:) = idxs(1:max_k);

            % was the correct label included for each top_k?
            for k = 1:numel(ks)
                judgments(iImg,k) = ismember(labels(iImg),top_k(iImg,1:ks(k)));
            end

            if iImg == 1
                labels(iImg)
                c2 = row_vector(c2)'
                idxs = row_vector(idxs)'
                judgments(iImg,:)
            end

        end

        varNames = [{'Images','Labels'} ...
                    arrayfun(@(x) sprintf('Top_%d',x), ks,'UniformOutput',0) ...
                    arrayfun(@(x) sprintf('Guess_%d',x), 1:max_k,'UniformOutput',0)];
        judTab = table(imgs,labels,judgments,top_k,'VariableNames',varNames);
        writetable(judTab,judFile);

        accTab = table;
        accTab.K = ks;
        accTab.Accuracy = mean(judgments)';
        writetable(accTab,accFile);

        fprintf('top k judgments computed\n');

    else
        fprintf('top k judgments found\n');

    end
end

function x = row_vector(x)
    if size(x,1) ~= numel(x)
        x = x';
    end
end
