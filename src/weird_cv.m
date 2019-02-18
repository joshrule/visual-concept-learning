function splits = weird_cv(ys,nPos,nRuns)
    splits = cell(size(ys,2),length(nPos),nRuns);

    for iClass = 1:size(ys,2)
        pos = find(ys(:,iClass));
        for iTrain = 1:length(nPos)
            for iRun = 1:nRuns
                ignored = sparse(size(ys,1),1);
                ignoredPos = pos(randperm(length(pos),length(pos)-nPos(iTrain)));
                ignored(ignoredPos) = 1;
                splits{iClass,iTrain,iRun} = logical(ignored);
            end
        end
    end
end
