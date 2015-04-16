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
            p.caching.nImgs,p.caching.hmaxHome,{});
        save(categoryFile,'rngState','categories','cachedCategories','imgList');
    end

    if ~exist([p.imgDir 'organicCache.flag'],'file')
        cacheImageNetCategoriesHelper(p.organicCatFile,p.organicImgDir)
        system(['touch ' p.imgDir 'organicCache.flag']);
    else
        fprintf('Organic C2 Caching Complete\n');
    end

    if ~exist([p.imgDir 'inorganicCache.flag'],'file')
        cacheImageNetCategoriesHelper(p.inorganicCatFile,p.inorganicImgDir)
        system(['touch ' p.imgDir 'inorganicCache.flag']);
    else
        fprintf('Inorganic C2 Caching Complete\n');
    end
end
