function [organicCategories,inorganicCategories] = choose5050Categories(organicImgDir,organicC3Dir,inorganicImgDir,inorganicC3Dir,patchSet,N)

    % construct C2 lists
    load([organicC3Dir 'splits.mat'],'trainFiles');
    restrictedOrganicCategories = listImageNetCategories(trainFiles);
    clear trainFiles;

    load([inorganicC3Dir 'splits.mat'],'trainFiles');
    restrictedInorganicCategories = listImageNetCategories(trainFiles);
    clear trainFiles;

    organicC2Files = restrictedCacheSearch(organicImgDir,patchSet,restrictedOrganicCategories,N);
    inorganicC2Files = restrictedCacheSearch(inorganicImgDir,patchSet,restrictedInorganicCategories,N);

    assert(sum(ismember(restrictedOrganicCategories,listImageNetCategories(organicC2Files))) == 0 && ...
           sum(ismember(restrictedInorganicCategories,listImageNetCategories(inorganicC2Files))) == 0, ...
	   'C3 vocabulary and Test Set overlap!');

    % construct c2 and labels
    allFiles = [organicC2Files; inorganicC2Files];
    allC2 = []; labels = [];
    for iClass = 1:2*N
        load(allFiles{iClass},'c2');
        allC2 = [allC2 c2];
        labels = blkdiag(labels, ones(1,size(c2,2)));
        clear c2;
    end
    c2 = allC2;
end

function files = restrictedCacheSearch(imgDir,patchSet,restrictedCategories,N)
    caches = dir([imgDir '*.' patchSet '.c2.mat']);
    allCategories = listImageNetCategories({caches.name});
    unrestrictedCategories = setdiff(allCategories,restrictedCategories);
    chosenCategories = unrestrictedCategories(randperm(length(unrestrictedCategories),N));
    files = strcat(imgDir,chosenCategories,'.',patchSet,'.c2.mat')';
end
