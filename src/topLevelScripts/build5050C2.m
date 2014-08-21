function [c2,labels,organicCategories,inorganicCategories,organicC2Files,inorganicC2Files] = build5050C2(organicImgDir,organicC3Dir,inorganicImgDir,inorganicC3Dir,patchSet,N,organicCategories,inorganicCategories)

    % construct C2 lists
    load([organicC3Dir 'splits.mat'],'trainFiles');
    restrictedOrganicCategories = listImageNetCategories(trainFiles);
    clear trainFiles;

    load([inorganicC3Dir 'splits.mat'],'trainFiles');
    restrictedInorganicCategories = listImageNetCategories(trainFiles);
    clear trainFiles;

    if (nargin < 8)
        organicCategories = restrictedCacheSearch(organicImgDir,patchSet,restrictedOrganicCategories,N);
        inorganicCategories = restrictedCacheSearch(inorganicImgDir,patchSet,restrictedInorganicCategories,N);
    end

    organicC2Files = strcat(organicImgDir, organicCategories, '.', patchSet, '.c2.mat')';
    inorganicC2Files = strcat(inorganicImgDir, inorganicCategories, '.', patchSet, '.c2.mat')';

    assert(sum(ismember(restrictedOrganicCategories,listImageNetCategories(organicC2Files))) == 0 && ...
           sum(ismember(restrictedInorganicCategories,listImageNetCategories(inorganicC2Files))) == 0, ...
	   'C3 vocabulary and Test Set overlap!');

    % construct c2 and labels
    allFiles = [organicC2Files; inorganicC2Files];
    allC2 = []; labels = [];
    for iClass = 1:(2*N)
        load(allFiles{iClass},'c2');
        allC2 = [allC2 c2];
        labels = blkdiag(labels, ones(1,size(c2,2)));
        clear c2;
    end
    c2 = allC2;
end

function cats = restrictedCacheSearch(imgDir,patchSet,restrictedCategories,N)
    caches = dir([imgDir '*.' patchSet '.c2.mat']);
    allCategories = listImageNetCategories({caches.name});
    unrestrictedCategories = setdiff(allCategories,restrictedCategories);
    cats = unrestrictedCategories(randperm(length(unrestrictedCategories),N));
end
