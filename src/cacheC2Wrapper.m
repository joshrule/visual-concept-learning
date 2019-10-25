function cacheC2Wrapper(imgs,featName,patchFile,hmaxHome,maxSize)
% cacheC2Wrapper(imgs,featName,patchFile,hmaxHome,maxSize)
%
% create C2 caches for the necessary ImageNet categories
%
% Args:
% - imgs: nImgs cell vector: the filenames of the images
% - featName: string, the name of the feature set
% - patchFile: string: the filename of the patch file
% - hmaxHome: string, the directory where HMAX-OCV is installed
% - maxSize: int, resize each image so this is the shortest edge length
    nImgs = size(imgs,1);
    fprintf('\tcomputing which of %d images still need caching with %s...\n',nImgs,patchFile);
    parfor iImg = 1:nImgs
        cacheFiles{iImg} = regexprep(imgs.file{iImg},'JPEG',[featName '_mat']);
        notCached(iImg) = ~exist(cacheFiles{iImg},'file');
    end
    cachesToMake = cacheFiles(find(notCached));
    imgsToCache = imgs.file(find(notCached));

    fprintf('\tcaching %d images with %s.\n',length(imgsToCache),patchFile);
    cacheC2Helper(patchFile,imgsToCache,cachesToMake,maxSize,hmaxHome);
end

function imgFiles = cacheC2Helper(patchFile,imgs,caches,maxSize,hmaxHome)
    idxs = [1:1000:length(imgs) length(imgs)+1];
    fprintf('\t0');
    for iPass = 1:(length(idxs)-1)
        start = idxs(iPass);
        stop = max(idxs(iPass+1)-1,1); 
        hmaxOCV(imgs(start:stop),patchFile,hmaxHome,maxSize);
        parfor iImg = start:stop
            [~,patchSet,~] = fileparts(patchFile);
            saveImg(imgs(iImg),patchSet,caches{iImg});
        end
        fprintf(', %d',stop);
    end
    fprintf('\n');
end

function saveImg(img,patchSet,cache)
    c2 = xmlC22matC2(img,patchSet);
    save(cache,'c2','-mat');
end
