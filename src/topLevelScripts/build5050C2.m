function [c2,labels,organicC2Files,inorganicC2Files] = build5050C2(organicImgDir,organicC3Dir,inorganicImgDir,inorganicC3Dir,patchSet,N)

% construct C2 lists
    load([organicC3Dir 'splits.mat'],'trainFiles');
    restrictedOrganicFiles = regexprep(regexprep(trainFiles,'josh/','joshrule/'),'c2Cache/','organicC2Cache/');
    load([inorganicC3Dir 'splits.mat'],'trainFiles');
    restrictedInorganicFiles = regexprep(trainFiles,'josh/','joshrule/');
    organicC2Files = restrictedCacheSearch(organicImgDir,patchSet,restrictedOrganicFiles,N);
    inorganicC2Files = restrictedCacheSearch(inorganicImgDir,patchSet,restrictedInorganicFiles,N);

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

function files = restrictedCacheSearch(imgDir,patchSet,restrictedFiles,N)
    caches = dir([imgDir '*.' patchSet '.c2.mat']);
    allFiles = strcat(imgDir,{caches.name});
    unrestrictedFiles = setdiff(allFiles,restrictedFiles);
    files = unrestrictedFiles(randperm(length(unrestrictedFiles),N));
end
