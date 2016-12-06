function evaluateFeatureSets(p,basetype,files,genMat,semMat)
% evaluateFeatureSets(p,basetype,files)

    % % % build all the stuff we need in advance % % %

    files_tr= files.file(strcmp(files.type,'training'));
    files_te= files.file(strcmp(files.type,'validation'));
    categories_tr = files.synset(strcmp(files.type,'training'));
    categories_te = files.synset(strcmp(files.type,'validation'));
    fprintf('training *and* testing image/category lists configured\n');

    for iFile = 1:length(files_tr)
        [d,f,~] = fileparts(files_tr{iFile});
        files_tr_ge{iFile} = [d '/' f '.' basetype '_gen_mat'];
        files_tr_ca{iFile} = [d '/' f '.' basetype '_cat_mat'];
    end
    fprintf('Built training general *and* conceptual feature cache lists\n');

    for iFile = 1:length(files_te)
        [d,f,~] = fileparts(files_te{iFile});
        files_te_ge{iFile} = [d '/' f '.' basetype '_gen_mat'];
        files_te_ca{iFile} = [d '/' f '.' basetype '_cat_mat'];
    end
    fprintf('Built testing general *and* conceptual feature cache lists\n');

    [m_tr_ge,labels_tr_ge] = buildActivationMatrix(files_tr_ge,categories_tr);
    [m_tr_ca,labels_tr_ca] = buildActivationMatrix(files_tr_ca,categories_tr);
    [m_te_ge,labels_te_ge] = buildActivationMatrix(files_te_ge,categories_te);
    [m_te_ca,labels_te_ca] = buildActivationMatrix(files_te_ca,categories_te);
    fprintf('Built all activations\n');

    assert(isequal(labels_tr_ge,labels_tr_ca), 'evaluation: cache failure!');
    labels_tr = labels_tr_ge;
    clear labels_tr_ge labels_tr_ca;
    fprintf('Deleted extra training label matrix\n');

    assert(isequal(labels_te_ge,labels_te_ca), 'evaluation: cache failure!');
    labels_te = labels_te_ge;
    clear labels_te_ge labels_te_ca;
    fprintf('Deleted extra testing label matrix\n');

    % We're training 1000 way classifiers, so we won't load 1, 2, 4,... images
    % total, but images/class. Let's say we have 20 splits. Then, we have 20 *
    % (1+2+4+8+16+...+512) = 20*1023 images whose similarities are needed for
    % each category, which is far more than the number of files on disk.
    % So, we want to load the scores matrices in advance.
% skipping general & semantics comparisons for now
%   general_similarity = genMat;
%   semantic_similarity = semMat;
%   fprintf('we have our similarity matrices\n');

    % create cross-validation splits for training with 1,2,4,8,... examples
    % (testing will always use the same images for comparison)
    if ~exist([p.outDir 'splits.mat'],'file')
        rngState = rng;
        cvsplit = multiclass_cv(labels_tr,p.nTrainingExamples,p.nRuns);
        save([p.outDir 'splits.mat'],'-mat','rngState','cvsplit');
        fprintf('50/50 splits generated\n\n');
    else
        load([p.outDir 'splits.mat'],'-mat','cvsplit');
        fprintf('50/50 splits loaded\n\n');
    end

    % create cross-validation splits for evaluating how effective each threshold is
    if ~exist([p.outDir 'featureSetSplits.mat'],'file') 
        nCategories = length(unique(categories_tr));
        rngState = rng;
        cvSource = randperm(nCategories);
        nPerCV = nCategories/p.nValidationRuns;
        count = 1;
        for iRun = 1:p.nValidationRuns
            featureSetSplit(iRun,:) = setdiff(1:nCategories,cvSource(count:(count+nPerCV-1)));
            count = count+nPerCV;
        end
        save([p.outDir 'featureSetSplits.mat'],'rngState','featureSetSplit');
        fprintf('feature set splits generated\n\n');
    else
        load([p.outDir 'featureSetSplits.mat'],'featureSetSplit');
        fprintf('feature set splits loaded\n\n');
    end

    % % % create the function that actually performs the evaluations % % %

    function evaluationHelper(outStem,m_tr,m_te,nFeats,type,scores)
        if nargin < 6, scores = m_tr; end;
        if nargin < 5, type = 'max'; end;
        if nargin < 4, nFeats = size(m_tr,1); end;
        outFile = [p.outDir outStem '-evaluation.mat'];
        if ~exist(outFile,'file')
            rngState = rng;
            [top1,top5,models,classVals,features] = evaluatePerformanceAlt( ...
              m_tr,labels_tr,m_te,labels_te,cvsplit,p.method,p.options, ...
              nFeats,[],type,scores);
            save(outFile,'-mat', 'rngState', 'top1', 'top5', 'models', ...
                'classVals', 'features');
        end
        fprintf('%s evaluated\n',outStem);
    end

    % % % restart the cluster % % %

%   delete(gcp);
%   cluster = parcluster('local');
%   cluster.NumWorkers=30;
%   parpool(cluster,30);
%   fprintf('initialized 30 thread parallel pool\n');

    % % % do the general feature comparisons % % %

    % evaluate all general features
    evaluationHelper([basetype '-general'],m_tr_ge,m_te_ge);

    % evaluate based on the general features above or below the threshold 'thresh'
    for iThresh = 1:length(p.nTestingThreshes)
        thresh = num2str(p.nTestingThreshes(iThresh)*max(max(m_tr_ge)));
        invScores = max(max(m_tr_ge)) - m_tr_ge;
        evaluationHelper([basetype '-above-thresh-' thresh],m_tr_ge,m_te_ge,inf,thresh);
        evaluationHelper([basetype '-below-thresh-' thresh],m_tr_ge,m_te_ge,inf,thresh,invScores);
    end
    validateFeatureSets(p,[basetype '-above-thresh'],featureSetSplit);
    validateFeatureSets(p,[basetype '-below-thresh'],featureSetSplit);

    % % % do the categorical feature comparisons % % %

    % evaluate all categorical features
    evaluationHelper([basetype '-categorical'],m_tr_ca,m_te_ca);

    % evaluate based on the categorical features above or below certain thresholds
    for iThresh = 1:length(p.nTestingThreshes)
        thresh = num2str(p.nTestingThreshes(iThresh));
        invScores = max(max(m_tr_ca)) - m_tr_ca;
        evaluationHelper([basetype '-thresh-normal-' thresh],m_tr_ca,m_te_ca,inf,thresh);
        evaluationHelper([basetype '-thresh-inverted-' thresh],m_tr_ca,m_te_ca,inf,thresh,invScores);
%       evaluationHelper([basetype '-thresh-by-semantics-' thresh],m_tr_ca,m_te_ca,inf,thresh,semantic_similarity);
    end
%   for iThresh = 1:length(p.nVisualThreshes)
%       thresh = num2str(p.nVisualThreshes(iThresh));
%       evaluationHelper([basetype '-thresh-by-visual-' thresh],m_tr_ca,m_te_ca,inf,thresh,general_similarity);
%   end
    validateFeatureSets(p,[basetype '-thresh-normal'],featureSetSplit);
    validateFeatureSets(p,[basetype '-thresh-inverted'],featureSetSplit);
%   validateFeatureSets(p,[basetype '-thresh-by-semantics'],featureSetSplit);
%   validateFeatureSets(p,[basetype '-thresh-by-visual'],featureSetSplit);

    % % % evaluate the combination of general and categorical features % % %

    evaluationHelper([basetype '-super'],[m_tr_ge;m_tr_ca],[m_te_ge;m_te_ca]);

end

function validateFeatureSets(p,type,splits)
    outFile = [p.outDir type '-validation.mat'];
    if ~exist(outFile,'file');
        dirInfo = dir([p.outDir type '*-evaluation.mat']);
        candidateFiles = strcat(p.outDir,{dirInfo.name});
        candidateTypes = regexprep(candidateFiles,[p.outDir type '-(?<n>.+)-evaluation\.mat'],'$<n>');
        for iType = 1:length(candidateTypes)
            load(candidateFiles{iType},'dprimes','features');
            for iSplit = 1:size(splits,1)
                trainingData(iSplit,iType,:,:,:) = dprimes(splits(iSplit,:),:,:);
                invSplit = setdiff(1:size(dprimes,1),splits(iSplit,:));
                possibleTestData(iSplit,iType,:,:,:) = dprimes(invSplit,:,:);
                for iTrain = 1:size(trainingData,4)
                    for iSplit2 = 1:size(trainingData,5) % n random splits
                        for iSplit3 = 1:length(invSplit)
                            possibleTestFeatures{iSplit,iType,iSplit3,iTrain,iSplit2} = features{invSplit(iSplit3),iTrain,iSplit2};
                            possibleTestFeatureSizes(iSplit,iType,iSplit3,iTrain,iSplit2) = length(possibleTestFeatures{iSplit,iType,iSplit3,iTrain,iSplit2});
                        end
                    end
                end
            end
        end
        % trainingData Dims = [validationSplits,thresholds,categories,nTrainingExamples,random splits]
        % collapse across categories collected
        collapsedData = squeeze(mean(trainingData,3));
        % top performance for each combination of validationSplit, and nTrainingExamples, taken over thresholds
        for iSplit = 1:size(collapsedData,1)
            for iTrain = 1:size(collapsedData,3)
                for iSplit2 = 1:size(collapsedData,4)
                    [topPerformance(iSplit,iTrain,iSplit2),topIdx(iSplit,iTrain,iSplit2)] = max(collapsedData(iSplit,:,iTrain,iSplit2));
                    chosenCandidate{iSplit,iTrain,iSplit2} = candidateTypes{topIdx(iSplit,iTrain,iSplit2)};
                    load(candidateFiles{topIdx(iSplit,iTrain,iSplit2)},'dprimes','features');
                    invSplit = setdiff(1:size(dprimes,1),splits(iSplit,:));
                    testData(iSplit,:,iTrain,iSplit2) = dprimes(invSplit,iTrain,iSplit2);
                    for iSplit3 = 1:length(invSplit)
                        testFeatures{iSplit,iSplit3,iTrain,iSplit2} = features{invSplit(iSplit3),iTrain,iSplit2};
                        testFeatureSizes(iSplit,iSplit3,iTrain,iSplit2) = length(testFeatures{iSplit,iSplit3,iTrain,iSplit2});
                    end
                end
            end
        end
        save([p.outDir type '-validation.mat'],'candidateFiles','candidateTypes',...
            'trainingData','collapsedData','topPerformance','topIdx', ...
            'chosenCandidate','testData','testFeatures','testFeatureSizes','possibleTestData','possibleTestFeatures','possibleTestFeatureSizes','-v7.3');
    end
    fprintf('%s validation complete\n',type);
end
