function [m,is] = compareClassifiersByTopFeatures(v1,v2,n)
    idxs1 = collectTopIndices(v1,n);
    idxs2 = collectTopIndices(v2,n);

    m = nan(size(idxs1,1),size(idxs2,1));
    is = cell(size(idxs1,1),size(idxs2,1));
    for iV1 = 1:size(idxs1,1)
        for iV2 = 1:size(idxs2,1)
            idxsUsed = union(idxs1(iV1,:),idxs2(iV2,:));
            m(iV1,iV2) = corr(v1(iV1,idxsUsed)',v2(iV2,idxsUsed)');
            is{iV1,iV2} = idxsUsed;
        end
    end
end

function idxs = collectTopIndices(m,n)
    [~,tmpIdxs] = sort(abs(m),2,'descend');
    idxs = tmpIdxs(:,1:n);
end
