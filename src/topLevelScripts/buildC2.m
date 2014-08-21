function [c2,labels] = buildC2(c2Files)
% [c2,labels] = buildC2(c2Files)
%
% construct c2 and labels
    allC2 = []; labels = [];
    for iClass = 1:(2*N)
        load(allFiles{iClass},'c2');
        allC2 = [allC2 c2];
        labels = blkdiag(labels, ones(1,size(c2,2)));
        clear c2;
    end
    c2 = allC2;
end
