function evaluateFeatureSets(p,c2Files,organicC3Files,inorganicC3Files)
% c3Evaluation(p,c2Files,organicC3Files,inorganicC3Files)

    if (nargin > 2)
        assert(isequal(listImageNetCategories(c2Files), ...
                       listImageNetCategories(organicC3Files)) && ...
               isequal(listImageNetCategories(c2Files), ...
                       listImageNetCategories(inorganicC3Files)), ...
               'evaluation: category listing failure');
    end

    % create activation matrices
    [c2,c2Labels] = buildC2(c2Files);
    basetype = regexprep(c2Files{1},'.+\.(?<type>.+)\.c2\.mat','$<type>');
    fprintf('Built C2 activations for basetype %s\n',basetype);
    if (nargin > 2)
        [oc3,oc3Labels] = buildC3(organicC3Files);
        [ic3,ic3Labels] = buildC3(inorganicC3Files);
        fprintf('Built organic C3 and inorganic C3 activations for basetype %s\n',basetype);

        assert(isequal(c2Labels,oc3Labels) && isequal(c2Labels,ic3Labels), ...
            'evaluation: cache failure!');
        cc3 = [oc3;ic3];
    end

    labels = c2Labels;

    % create cross-validation splits
    if ~exist([p.outDir 'splits.mat'],'file')
        rngState = rng;
        cvsplit = cv(labels,p.nTrainingExamples,p.nRuns);
        save([p.outDir 'splits.mat'],'rngState','cvsplit');
        fprintf('50/50 splits generated\n\n');
    else
        load([p.outDir 'splits.mat'],'cvsplit');
        fprintf('50/50 splits loaded\n\n');
    end

    if p.nTestingCategories > 40
        if ~exist([p.outDir 'featureSetSplits.mat'],'file') 
            rngState = rng;
            for iRun = 1:p.nRuns
                featureSetSplit(iRun,:) = randperm(length(c2Files),80);
            end
            save([p.outDir 'featureSetSplits.mat'],'rngState','featureSetSplit');
            fprintf('feature set splits generated\n\n');
        else
            load([p.outDir 'featureSetSplits.mat'],'featureSetSplit');
            fprintf('feature set splits loaded\n\n');
        end
    end

    function evaluationHelper(outStem,m,nFeats,type,scores)
        if nargin < 5, scores = m; end;
        if nargin < 4, type = 'max'; end;
        if nargin < 3, nFeats = size(m,1); end;
        outFile = [p.outDir outStem '-evaluation.mat'];
        if ~exist(outFile,'file')
            rngState = rng;
            [aucs,dprimes,models,classVals,features] = evaluatePerformance( ...
              m,labels,cvsplit,p.method,p.options,nFeats,[],type,scores);
            save(outFile,'rngState','labels','m','p','aucs', ...
              'dprimes', 'models', 'classVals', 'features', '-v7.3');
        end
        fprintf('%s evaluated\n',outStem);
    end

    % evaluate all C2 features
    evaluationHelper(basetype,c2);
    % evaluate based on those C2 features above (or below) the threshold 'thresh'
    for iThresh = 1:length(p.nTestingThreshes)
        thresh = p.nTestingThreshes(iThresh)*max(max(c2));
        invScores = max(max(c2)) - c2;
        evaluationHelper([basetype '-below-thresh-' num2str(thresh)],c2,inf,num2str(thresh),invScores);
    end
    % evaluate based on the top (or bottom or random) 'nFeats' C2 features 
    for iFeat = 1:length(p.nTestingFeats)
        nFeats = p.nTestingFeats(iFeat);
    end
    % validation
    if p.nTestingCategories > 40
        validateFeatureSets(p,[basetype '-below-thresh'],featureSetSplit);
    end

    % C3 Feature comparisons
    if (nargin > 2)
        if exist([p.outDir 'semantic-similarities.mat'],'file') && exist([p.outDir 'visual-similarities.mat'],'file')
            load([p.outDir 'semantic-similarities.mat'],'fullVocabVsTestSemanticSimilarities');
            load([p.outDir 'visual-similarities.mat'],'fullVocabVsTestVisualSimilarities');
            sims{1,1} = fullVocabVsTestSemanticSimilarities;
            sims{1,2} = fullVocabVsTestVisualSimilarities;
        end
        types = {'combined'};
        mats = {cc3};

        for iType = 1:length(types)
            evaluationHelper([basetype '-' types{iType}],mats{iType});
            evaluationHelper([basetype '-' types{iType} '-super'],[mats{iType};c2]);
            for iThresh = 1:length(p.nTestingThreshes)
                thresh = p.nTestingThreshes(iThresh);
                invScores = max(max(mats{iType})) - mats{iType};
                evaluationHelper([basetype '-' types{iType} '-thresh-' num2str(thresh)],mats{iType},inf,num2str(thresh));
                evaluationHelper([basetype '-' types{iType} '-thresh-inverted-' num2str(thresh)],mats{iType},inf,num2str(thresh),invScores);
                if exist('sims','var')
                    evaluationHelper([basetype '-' types{iType} '-thresh-by-semantics-' num2str(thresh)],mats{iType},inf,num2str(thresh),sims{iType,1});
                end
            end
            for iThresh = 1:length(p.nVisualThreshes)
                thresh = p.nVisualThreshes(iThresh);
                if exist('sims','var')
                    evaluationHelper([basetype '-' types{iType} '-thresh-by-visual-' num2str(thresh)],mats{iType},inf,num2str(thresh),sims{iType,2});
                end
            end
            if p.nTestingCategories > 40
                validateFeatureSets(p,[basetype '-' types{iType} '-thresh-'],featureSetSplit);
                validateFeatureSets(p,[basetype '-' types{iType} '-thresh-inverted-'],featureSetSplit);
                if exist('sims','var')
                    validateFeatureSets(p,[basetype '-' types{iType} '-thresh-by-semantics-'],featureSetSplit);
                    validateFeatureSets(p,[basetype '-' types{iType} '-thresh-by-visual-'],featureSetSplit);
                end
            end
        end
    end
end

function validateFeatureSets(p,type,splits)
    outFile = [p.outDir type '-validation.mat'];
    if ~exist(outFile,'file');
        dirInfo = dir([p.outDir type '*-evaluation.mat']);
        candidateFiles = strcat(p.outDir,{dirInfo.name});
        candidateTypes = regexprep(candidateFiles,[p.outDir type '-(?<n>.+)-evaluation\.mat'],'$<n>');
        for iType = 1:length(candidateTypes)
            load(candidateFiles{iType},'dprimes');
            for iSplit = 1:size(splits,1)
                trainingData(iSplit,iType,:,:) = squeeze(dprimes(splits(iSplit,:),1,:));
            end
        end
        % collapse across dprime splits and categories collected
        collapsedData = squeeze(mean(mean(trainingData,4),3));
        % top performance for each split
        [topPerformance,topIdx] = max(collapsedData,[],2);
        for iSplit = 1:size(splits,1)
            chosenCandidate{iSplit} = candidateTypes{topIdx(iSplit)};
            load(candidateFiles{topIdx(iSplit)},'dprimes');
            invSplit = setdiff(1:size(dprimes,1),splits(iSplit,:));
            testData(iSplit,:,:,:) = dprimes(invSplit,:,:);
        end
        save([p.outDir type '-validation.mat'],'candidateFiles','candidateTypes',...
            'trainingData','collapsedData','topPerformance','topIdx', ...
            'chosenCandidate','testData');
    end
    fprintf('%s validation complete\n',type);
end
