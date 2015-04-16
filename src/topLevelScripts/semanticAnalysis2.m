function semanticAnalysis2(p)
    load([p.home 'evaluation/5050v' p.suffix '/c3Vocabulary.mat'],'organicC3Vocab','inorganicC3Vocab');
    load([p.home 'evaluation/5050v' p.suffix '/chosenCategories.mat'],'organicCategories','inorganicCategories');
    o = organicC3Vocab;
    i = inorganicC3Vocab;
    c = [o;i];
    c2Cats = [organicCategories inorganicCategories]';

    %semanticHelper(o,[p.home 'organic-evaluation.mat'],[p.home 'organic-semantic-analysis.mat']);
    %semanticHelper(i,[p.home 'inorganic-evaluation.mat'],[p.home 'inorganic-semantic-analysis.mat']);
    semanticHelper(c,[p.home 'evaluation/5050v' p.suffix '/combined-thresh-0.55-evaluation.mat'],[p.home 'evaluation/5050v' p.suffix '/combined-thresh-0.55-semantic-analysis.mat']);

    function semanticHelper(wnids,filename,outfile)
        load(filename,'features');
        [nClass,nTrain,nSplit] = size(features);
        [d,f,e] = fileparts(outfile);
        for iClass = 1:nClass
            for iTrain = 1:nTrain
                for iSplit = 1:nSplit
                    fprintf('1: %d %d %d/%d %d %d\n',iClass,iTrain,iSplit,nClass,nTrain,nSplit);
                    f2 = [f '-' num2str(iClass) '-' num2str(iTrain) '-' num2str(iSplit)];
                    individualAnalysis(p,features{iClass,iTrain,iSplit},c2Cats{iClass},wnids,[d '/' f2 e]);
                end
            end
        end
    end
end

function analysis = individualAnalysis(p,features,cat,wnids,outfile)
    if ~exist(outfile,'file')
        analysis.features = features;
        analysis.name = imageNetName(cat,p.srFile);
        n = length(features);
        analysis.nFeats = n;
        analysis.chosenNames = imageNetNames(wnids(analysis.features),p.srFile);
        unchosenFeatures = setdiff(1:length(wnids),analysis.features);
        analysis.selectedUnchosenFeatures = unchosenFeatures(randi(length(unchosenFeatures),1,n));
        analysis.pairwiseChosenScores = pairwiseScores(p.simFile,{cat},wnids(analysis.features));
        analysis.pairwiseUnchosenScores = pairwiseScores(p.simFile,{cat},wnids(analysis.selectedUnchosenFeatures));
        analysis.meanChosenScores = mean(analysis.pairwiseChosenScores);
        analysis.meanUnchosenScores = mean(analysis.pairwiseUnchosenScores);
        save(outfile,'analysis','-v7.3');
    end
end

function names = imageNetNames(wnids,srFile)
    names = cell(length(wnids),1);
    parfor i = 1:length(wnids)
        names{i} = imageNetName(wnids{i},srFile);
    end
end

function name = imageNetName(wnid,srFile)
    w = wnidToDefinition(srFile,wnid);
    name = w.words;
end

function semanticScore = pairwiseScores(simFile,wnids1,wnids2)
    semanticScore = nan(length(wnids1),length(wnids2));
    for i1 = 1:length(wnids1)
        parfor i2 = 1:length(wnids2)
            semanticScore(i1,i2) = str2num(perl(simFile,wnids1{i1},wnids2{i2}));
        end
    end
end
