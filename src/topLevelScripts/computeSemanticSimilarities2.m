function setupSemanticSimilarities(p,c2Files,organicC3Files,inorganicC3Files)
% setupSemanticSimilarities(p,c2Files,organicC3Files,inorganicC3Files)
    mf = matfile([p.outDir 'semantic-similarities.mat'],'Writable',true);

    c2 = buildC2(c2Files);
    [oc3,olabels,oImageFiles] = buildC3(organicC3Files);
    [ic3,ilabels,iImageFiles] = buildC3(inorganicC3Files);
    c3 = [oc3; ic3];
    assert(isequal(ilabels,olabels), 'Failed label equality in semantic analysis 1\n');
    assert(isequal(iImageFiles,oImageFiles), 'Failed image list equality in semantic analysis 1\n');
    clear ilabels olabels oImageFiles iImageFiles oc3 ic3;

    oC3Cats = load([p.outDir 'organic-categories.mat'],'c3Categories');
    iC3Cats = load([p.outDir 'inorganic-categories.mat'],'c3Categories');
    vocabCategories = [reshape(oC3Cats.c3Categories,[],1); reshape(iC3Cats.c3Categories,[],1)];
    initializeMFValue(mf,'vocabCategories',vocabCategories);
    initializeMFValue(mf,'testCategories',listImageNetCategories(c2Files));
    initializeMFValue(mf,'testVsTestSemanticSimilarities',nan(length(testCategories),length(testCategories));
    initializeMFValue(mf,'vocabVsTestSemanticSimilarities',nan(length(vocabCategories),length(testCategories));
    initializeMFValue(mf,'vocabVsVocabSemanticSimilarities',nan(length(vocabCategories),length(vocabCategories));
    initializeMFValue(mf,'fullVocabVsTestSemanticSimilarities',nan(length(vocabCategories),size(c2,2));

    mf.testVsTestSemanticSimilaritiesf = @(imf,i,j) pairwiseScore(imf,'testVsTestSemanticSimilarities',testCategories{i},testCategories{j},i,j);
    mf.vocabVsTestSemanticSimilaritiesf = @(imf,i,j) pairwiseScore(imf,'vocabVsTestSemanticSimilarities',vocabCategories{i},testCategories{j},i,j);
    mf.vocabVsVocabSemanticSimilaritiesf = @(imf,i,j) pairwiseScore(imf,'testVsTestSemanticSimilarities',vocabCategories{i},vocabCategories{j},i,j);

    pairwiseScores(mf,'testVsTestSemanticSimilarities',testCategories,testCategories);
    pairwiseScores(mf,'vocabVsTestSemanticSimilarities',vocabCategories,testCategories);

    for i = 1:length(vocabCategories)
        for j = 1:size(c2,2)
            if isnan(mf.fullVocabVsTestSemanticSimilarities(i,j))
                mf.fullVocabVsTestSemanticSimilarities(i,j) = ...
                  vocabVsTestSimilarities(i,ceil(j/p.nImgs));
            end
        end
    end
end

function pairwiseScores(mf,var,wnids1,wnids2)
    simFile = '/data/josh/ruleRiesenhuber2013/src/external/rwSimilarity.pl';
    simDir = '/data/josh/ruleRiesenhuber2013/imageSets/imageNet/semanticSimilarities/';
    [idxIs,idxJs] = meshgrid(1:length(wnids1),1:length(wnids2));
    parfor i1 = 1:numel(idxIs)
        if ~exist([simDir wnids1{idxIs(i1)} '.' wnids2{idxJs(i1)} '.similarity'],'file')
            perl(simFile,wnids1{idxIs(i1)},wnids2{idxJs(i1)},simDir);
        end
    end
    fprintf('-- finished %d remaining\n',numel(idxIs));
    existingVars = whos('-file',outFile);
    if ~ismember(var,{existingVars.name})
        fprintf('writing to a single file\n');
        for i1 = 1:length(wnids1)
            for i2 = 1:length(wnids2)
                data = load([simDir wnids1{i1} '.' wnids2{i2} '.similarity']);
                mf.(var)(i1,i2) = data;
            end
        end
        fprintf('cached %s\n',var);
    end
end

function pairwiseScore(mf,var,wnid1,wnid2,i,j)
    simFile = '/data/josh/ruleRiesenhuber2013/src/external/rwSimilarity.pl';
    simDir = '/data/josh/ruleRiesenhuber2013/imageSets/imageNet/semanticSimilarities/';
    if ~exist([simDir wnid1 '.' wnid2 '.similarity'],'file')
        perl(simFile,wnid1,wnid2,simDir);
    end
    data = load([simDir wnid1 '.' wnid2 '.similarity']);
    mf.(var)(i,j) = data;
end
