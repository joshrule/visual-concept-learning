function cacheC2Wrapper(p)
% cacheC2Wrapper(p)
%
% create C2 caches for the necessary ImageNet categories
%
% p: struct, the parameters set in the top-level script
    cacheImageNetCategoriesHelper(p,'organic');
    cacheImageNetCategoriesHelper(p,'inorganic');
end

function cacheImageNetCategoriesHelper(p,type)
    categoryFile = getfield(p,[type 'CatFile']);
    imgDir = getfield(p,[type 'ImgDir']);
    for iSet = 1:length(p.caching.patchSets)
        patchSet = p.caching.patchSets{iSet};
        patchFile = [p.caching.patchDir patchSet p.caching.patchExt];
        flagFile = [p.imgDir type '-' patchSet '-cache.flag'];
        if ~exist(flagFile,'file')
            load(categoryFile,'categories')
            rngState = rng;
            [cachedCategories{iSet},imgList{iSet}] = cacheWithoutDuplicates(...
                p.imgHome,imgDir,categories,patchFile,p.caching.maxSize, ...
                p.caching.nImgs,p.caching.hmaxHome,{});
            system(['touch ' flagFile]);
            save(categoryFile,'rngState','categories','cachedCategories','imgList');
            fprintf('%s-%s C2 caching complete\n',type,patchSet);
        else
            fprintf('%s-%s C2 caching previously completed\n',type,patchSet);
        end
    end
end
