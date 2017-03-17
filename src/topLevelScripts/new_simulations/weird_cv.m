function splits = weird_cv(ys,nPos,nRuns)
    splits = cell(size(ys,2),length(nPos),nRuns);

    for iClass = 1:size(ys,2)
        y = ys(:,iClass);
        for iTrain = 1:length(nPos)
            for iRun = 1:nRuns
                pos = find(y);
                neg = find(~logical(y));
                n = nPos(iTrain);
                chosenPos = pos(randperm(length(pos),n));
                
                tmp = zeros(size(y));
                tmp([chosenPos(:); neg(:)]) = 1;

                splits{iClass,iTrain,iRun} = logical(tmp);
            end
        end
    end
end
