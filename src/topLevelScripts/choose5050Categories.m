function [organicC2Files,inorganicC2Files] = choose5050Categories(organicImgDir,organicC3File,inorganicImgDir,inorganicC3File,patchSet,N)
    load(organicC3File,'organicFiles');
    restrictedOrganicFiles = listImageNetCategories(organicFiles);

    load(inorganicC3File,'inorganicFiles');
    restrictedInorganicFiles = listImageNetCategories(inorganicFiles);

    organicC2Files = restrictedCacheSearch(organicImgDir,patchSet,restrictedOrganicFiles,N);
    inorganicC2Files = restrictedCacheSearch(inorganicImgDir,patchSet,restrictedInorganicFiles,N);
end

function files = restrictedCacheSearch(imgDir,patchSet,restrictedCategories,N)
    caches = dir([imgDir '*.' patchSet '.c2.mat']);
    allCategories = listImageNetCategories({caches.name});
    unrestrictedCategories = setdiff(allCategories,restrictedCategories);
    allFiles = strcat(imgDir,unrestrictedCategories,['.' patchSet '.c2.mat']);
    files = allFiles(randperm(length(allFiles),N));
end
