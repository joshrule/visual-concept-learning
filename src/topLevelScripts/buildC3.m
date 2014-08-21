function [c3,labels,imageFiles] = buildC3(c3Files)
% [c3,labels,imageFiles] = buildC3(c3Files)
%
% construct c3 and labels
    allC3 = []; labels = []; imageFiles = [];
    for iClass = 1:length(c3Files)
        load(c3Files{iClass},'c3','imgFiles');
        allC3 = [allC3 c3];
	imageFiles = [imageFiles; imgFiles];
        labels = blkdiag(labels, ones(1,size(c3,2)));
        clear c3;
    end
    c3 = allC3;
end
