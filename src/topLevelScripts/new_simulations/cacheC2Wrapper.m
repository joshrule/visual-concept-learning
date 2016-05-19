function cacheC2Wrapper(imgs,featDir,patchFiles,hmaxHome,maxSize)
% cacheC2Wrapper(imgs,featDir,patchFiles,hmaxHome,maxSize)
%
% create C2 caches for the necessary ImageNet categories
    nImgs = size(imgs,1);
    for iType = 1:length(patchFiles)
        fprintf('\tdetermining which of %d images still need caching with %s...',nImgs,patchFiles{iType});
        notCached = nan(nImgs,1);
        cacheFiles = cell(nImgs,1);
        [~,patchSet,~] = fileparts(patchFiles{iType});

        parfor iImg = 1:nImgs
            synset = imgs.synset{iImg};
            [~,file,~] = fileparts(imgs.file{iImg});
            cacheDir = ensureDir([featDir patchSet '/' synset '/']);
            cacheFiles{iImg} = [cacheDir file '.mat'];
            notCached(iImg) = ~exist(cacheFiles{iImg},'file');
        end

        cachesToMake = cacheFiles(find(notCached));
        imgsToCache = imgs.file(find(notCached));
        fprintf('%d images\n',length(imgsToCache));

        fprintf('\tcaching %d images with %s.\n',length(imgsToCache),patchFiles{iType});
        cacheC2Helper(patchFiles{iType},imgsToCache,cachesToMake,maxSize,hmaxHome);
    end
end

function imgFiles = cacheC2Helper(patchFile,imgs,caches,maxSize,hmaxHome)
    idxs = 1:1000:length(imgs);
    if (idxs(end) ~= length(imgs))
        idxs = [idxs length(imgs)];
    end;
    fprintf('\t0');
    for iPass = 1:(length(idxs)-1)
        start = idxs(iPass);
        stop = idxs(iPass+1)-1; 
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
    save(cache,'c2');
end
