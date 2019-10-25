function cacheC3Wrapper(c3Files,c2Files,modelDir)
% cacheC3Wrapper(c3Files,c2Files,modelDir)
%
% Create C3 caches for the necessary ImageNet categories.
%
% Args:
% - c3Files: cell vector, a list of files to which to write C3 activations
% - c2Files: cell vector, a list of files from which to load C2 activations
% - modelDir: string, directory containing the C3 models
    nImgs = length(c3Files);
    fprintf('\tcomputing which of %d images still need caching with %s...\n',nImgs,modelDir);
    parfor iImg = 1:nImgs
        notCached(iImg) = ~exist(c3Files{iImg},'file');
    end
    cachesToMake = c3Files(find(notCached));
    imgsToCache = c2Files(find(notCached));

    fprintf('\tcaching %d images with %s.\n',length(imgsToCache),modelDir);
    cacheC3Helper(modelDir,imgsToCache,cachesToMake);
end

function cacheC3Helper(modelDir,imgs,caches)
    modelFile = [modelDir 'models.mat'];
    paramFile = [modelDir 'setup.mat'];
    c3models = load([modelDir 'models.mat'],'models');
    models = c3models.models;

    idxs = [1:1000:length(imgs) length(imgs)+1];
    fprintf('\t0');
    for iPass = 1:(length(idxs)-1)
        start = idxs(iPass);
        stop = max(idxs(iPass+1)-1,1); 
        c3 = cacheC3(imgs(start:stop),modelFile,paramFile,models)';
        parfor iImg = start:stop
            saveImg(c3(iImg-start+1,:),modelFile,paramFile,caches{iImg});
        end
        fprintf(', %d',stop);
    end
    fprintf('\n');
end

function saveImg(c2,modelFile,paramFile,outFile)
    save(outFile,'c2','modelFile','paramFile','-mat');
end
