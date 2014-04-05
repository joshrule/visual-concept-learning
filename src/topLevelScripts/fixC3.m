function fixC3(wnid,params)
    % set the stream for repeatable results
    RandStream.setGlobalStream(RandStream('mcg16807','Seed',0))

    % set some hacked together parameters
    home = '/home/josh/data/ruleRiesenhuber2013/';
    addpath(genpath([home 'src/']));
    imgDir = [home 'imageSets/imageNet/'];
    outDir = [home 'patchSets/fixC3/'];
    method = 'svm';
    options = struct('svmTrainFlags', '-s 0 -t 0 -c 0.1 -b 1 -q', ...
                     'svmTestFlags', '-b 1', ...
                     'alpha', 0.7, ...
                     'startPerIter', 200, ...
                     'threshold', 0.25, ...
                     'ratio', 1.0, ...
		     'div', mat2cell(params,size(params,1),ones(size(params,2),1)));

    parfor iParam = 1:length(params)
    	fixHelper(options(iParam),outDir,imgDir,method,wnid);
        fprintf('done with round %d/%d\n',iParam,length(params));
    end
end

function fixHelper(options,outDir,imgDir,method,wnid)
    % generate a new set of C3 units
    load([outDir '../oldC3Categories.mat'],'inorganicFiles'); fprintf('files...');
    [c2,labels] = responsesFromCaches(inorganicFiles,'c2');   fprintf('c2...');
    models = trainC3(c2,labels,method,options);               fprintf('c3...');
    type = ['test1Div' num2str(options.div)];
    save([outDir type '.mat'],'organicFiles','method','options','models','-v7.3');
    clear models;

    % cache C3 activations for our category
    c2File = [imgDir wnid '.kmeans.c2.mat'];
    c3File = [imgDir wnid '.' type '.c3.mat'];
    patchFile = [outDir type '.mat'];
    load(patchFile,'models');
    if ~exist(c3File,'file')
        cacheC3(c3File,c2File,patchFile,patchFile,models);
    end
   	clear models;
end
