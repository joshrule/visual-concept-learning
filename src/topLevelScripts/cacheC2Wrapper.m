function cacheC2Wrapper(p)
% cacheC2Wrapper(p)
%
% create C2 caches for the necessary ImageNet categories
%
% p: struct, the parameters set in the top-level script

    function cacheImageNetCategoriesHelper(categoryFile,imgDir)
        load(categoryFile,'categories')
        rngState = rng;
        [cachedCategories,imgList] = cacheWithoutDuplicates(p.imgHome, ...
            imgDir,categories,p.caching.patchFile,p.caching.maxSize, ...
            p.caching.nImgs,p.caching.hmaxHome,[]);
        save(categoryFile,'rngState','categories','cachedCategories','imgList');
    end

    cacheImageNetCategoriesHelper(p.organicCatFile,p.organicImgDir)
    cacheImageNetCategoriesHelper(p.inorganicCatFile,p.inorganicImgDir)
    fprintf('C2 Caching Complete\n');
end
