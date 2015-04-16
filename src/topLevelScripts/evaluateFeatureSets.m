function evaluateFeatureSets(p,c2Files,organicC3Files,inorganicC3Files)
% c3Evaluation(p,c2Files,organicC3Files,inorganicC3Files)
    assert(isequal(listImageNetCategories(c2Files), ...
                   listImageNetCategories(organicC3Files)) && ...
           isequal(listImageNetCategories(c2Files), ...
                   listImageNetCategories(inorganicC3Files)), ...
           'evaluation: category listing failure');

    % load activations
    [c2,c2Labels] = buildC2(c2Files);
    [oc3,oc3Labels] = buildC3(organicC3Files);
    [ic3,ic3Labels] = buildC3(inorganicC3Files);
    fprintf('Built C2, organic C3, and inorganic C3 activations\n');

    assert(isequal(c2Labels,oc3Labels) && isequal(c2Labels,ic3Labels), ...
        'evaluation: cache failure!');
    labels = c2Labels;
    cc3 = [oc3;ic3];

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

    function c3EvaluationHelper(outStem,m,nFeats,type)
        if nargin < 4, type = 'max'; end;
        if nargin < 3, nFeats = size(m,1); end;
        outFile = [p.outDir outStem '-evaluation.mat'];
        if ~exist(outFile,'file')
            rngState = rng;
            [aucs,dprimes,models,classVals,features] = evaluatePerformance(m,labels,cvsplit,p.method, ...
              p.options,nFeats,[],type);
            save(outFile,'rngState','labels','m','p','aucs', ...
              'dprimes', 'models', 'classVals', 'features', '-v7.3');
            clear rngState m aucs dprimes models classVals features;
        end
        fprintf('%s evaluated\n',outStem);
    end

    featsFile = [p.outDir 'sortedC3Features.mat'];
    if ~exist(featsFile,'file')
        nCats = 2*(p.nCategories)
        for iClass = 1:nCats
            [~,oIdx(iClass,:)] = sort(mean(oc3(:,find(labels(iClass,:))),2),'descend');
            [~,iIdx(iClass,:)] = sort(mean(ic3(:,find(labels(iClass,:))),2),'descend');
            [~,cIdx(iClass,:)] = sort(mean(cc3(:,find(labels(iClass,:))),2),'descend');
        end
        save([p.outDir 'sortedC3Features.mat'],'oIdx','iIdx','cIdx');
        fprintf('sorted C3 feature indices generated\n\n');
    else
        load([p.outDir 'sortedC3Features.mat'],'oIdx','iIdx','cIdx');
        fprintf('sorted C3 feature indices loaded\n\n');
    end

    c3EvaluationHelper('kmeans',c2);
    c3EvaluationHelper('organic',oc3);
    c3EvaluationHelper('organic-super',[oc3;c2]);
    c3EvaluationHelper('inorganic',ic3);
    c3EvaluationHelper('inorganic-super',[ic3;c2]);
    c3EvaluationHelper('combined',cc3);
    c3EvaluationHelper('combined-super',[cc3;c2]);
    for iThresh = 1:length(p.nTestingThreshes)
        thresh = p.nTestingThreshes(iThresh);
        c3EvaluationHelper(['organic-thresh-' num2str(thresh)],oc3,inf,num2str(thresh));
        c3EvaluationHelper(['inorganic-thresh-' num2str(thresh)],ic3,inf,num2str(thresh));
        c3EvaluationHelper(['combined-thresh-' num2str(thresh)],cc3,inf,num2str(thresh));
    end
    for iFeat = 1:length(p.nTestingFeats)
        nFeats = p.nTestingFeats(iFeat);
        c3EvaluationHelper(['organic-max-' num2str(nFeats)],oc3,nFeats,'max');
        c3EvaluationHelper(['inorganic-max-' num2str(nFeats)],ic3,nFeats,'max');
        c3EvaluationHelper(['combined-max-' num2str(nFeats)],cc3,nFeats,'max');
        c3EvaluationHelper(['kmeans-min-' num2str(nFeats)],c2,nFeats,'min');
        c3EvaluationHelper(['kmeans-max-' num2str(nFeats)],c2,nFeats,'max');
    end
end
