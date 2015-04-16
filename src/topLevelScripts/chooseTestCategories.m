function [organicCategories,inorganicCategories,organicC2Files,inorganicC2Files] = chooseTestCategories(p)
    if ~exist([p.outDir 'chosenCategories.mat'],'file')
        rngState = rng;
        [organicCategories,inorganicCategories,organicC2Files,inorganicC2Files] = ...
          chooseCategories(p.organicImgDir,p.organicC3Dir,p.inorganicImgDir,p.inorganicC3Dir, ...
            p.patchSet,p.nCategories);
        save([p.outDir 'chosenCategories.mat'],'rngState','organicCategories','inorganicCategories');
        save([p.outDir 'chosenC2Files.mat'],'rngState','organicC2Files','inorganicC2Files');
        fprintf('categories chosen!\n');
    else
        load([p.outDir 'chosenCategories.mat'],'organicCategories','inorganicCategories');
        load([p.outDir 'chosenC2Files.mat'],'organicC2Files','inorganicC2Files');
        fprintf('categories loaded!\n');
    end
end


function [organicCategories,inorganicCategories,organicC2Files,inorganicC2Files] = chooseCategories(organicImgDir,organicC3Dir,inorganicImgDir,inorganicC3Dir,patchSet,N)

    % construct C2 lists
    load([organicC3Dir 'setup.mat'],'files');
    restrictedOrganicCategories = listImageNetCategories(files);
    clear files;

    load([inorganicC3Dir 'setup.mat'],'files');
    restrictedInorganicCategories = listImageNetCategories(files);
    clear files;

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
