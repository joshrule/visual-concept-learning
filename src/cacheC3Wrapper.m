function cacheC3Wrapper(c3Files,c2Files,modelDir)
% cacheC3Wrapper(c3Files,c2Files,modelDir)
%
% create C3 caches for the necessary ImageNet categories
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

% function cacheC3Wrapper(c3Files,c2Files,modelDir)
%     c3models = load([modelDir 'models.mat'],'models');
%     models = c3models.models;
%     for i = 1:length(c3Files)
%         if mod(i,1000) == 0
%             fprintf('%d/%d',i, length(c3Files));
%         end
%         if ~exist(c3Files{i},'file')
%             cacheC3(c3Files{i},c2Files{i}, ...
%               [modelDir 'models.mat'],[modelDir 'setup.mat'],models);
%             fprintf('%d: cached %s\n',i,c3Files{i});
%         else
%             fprintf('%d: found %s\n',i,c3Files{i});
%         end
%     end
% end
