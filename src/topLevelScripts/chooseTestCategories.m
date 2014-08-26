function [organicCategories,inorganicCategories,organicC2Files,inorganicC2Files] = chooseTestCategories(p)
    if ~exist([p.outDir 'chosenCategories.mat'],'file')
        rngState = rng;
        [organicCategories,inorganicCategories,organicC2Files,inorganicC2Files] = ...
          chooseCategories(p.organicImgDir,p.organicC3Dir,p.inorganicImgDir,p.inorganicC3Dir, ...
            p.patchSet,p.nCategories);
        save([outDir 'chosenCategories.mat'],'rngState','organicCategories','inorganicCategories');
        save([outDir 'chosenC2Files.mat'],'rngState','organicC2Files','inorganicC2Files');
    else
        load([outDir 'chosenCategories.mat'],'organicCategories','inorganicCategories');
        load([outDir 'chosenC2Files.mat'],'organicC2Files','inorganicC2Files');
    end
end


function [organicCategories,inorganicCategories,organicC2Files,inorganicC2Files] = chooseCategories(organicImgDir,organicC3Dir,inorganicImgDir,inorganicC3Dir,patchSet,N)

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
end

function cats = restrictedCacheSearch(imgDir,patchSet,restrictedCategories,N)
    caches = dir([imgDir '*.' patchSet '.c2.mat']);
    allCategories = listImageNetCategories({caches.name});
    unrestrictedCategories = setdiff(allCategories,restrictedCategories);
    cats = unrestrictedCategories(randperm(length(unrestrictedCategories),N));
end
