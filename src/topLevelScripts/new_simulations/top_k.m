function topK = top_k(classificationValues,labels,k)
    [nExamples,nClasses] = size(labels);
    topKs = nan(nExamples,1);
    parfor item = 1:nExamples
        [~,idxs] = sort(classificationValues(item,:),'descend');
        topKs(item) = ismember(find(labels(item,:)),idxs(1:k));
    end
    topK = mean(topKs);
end
