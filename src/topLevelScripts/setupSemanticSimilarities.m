function setupSemanticSimilarities(p,c2Files,organicC3Files,inorganicC3Files)
% setupSemanticSimilarities(p,c2Files,organicC3Files,inorganicC3Files)
    outFile = [p.outDir 'semantic-similarities.mat'];
    if ~exist([outFile '.flag'],'file')
        c2 = buildC2(c2Files);
        [oc3,olabels,oImageFiles] = buildC3(organicC3Files);
        [ic3,ilabels,iImageFiles] = buildC3(inorganicC3Files);
        c3 = [oc3; ic3];
        assert(isequal(ilabels,olabels), 'Failed label equality in semantic analysis 1\n');
        assert(isequal(iImageFiles,oImageFiles), 'Failed image list equality in semantic analysis 1\n');
        oC3Cats = load([p.outDir 'organic-categories.mat'],'c3Categories');
        iC3Cats = load([p.outDir 'inorganic-categories.mat'],'c3Categories');
        vocabCategories = [reshape(oC3Cats.c3Categories,[],1); reshape(iC3Cats.c3Categories,[],1)];
        testCategories = listImageNetCategories(c2Files);
        clear ilabels olabels oImageFiles iImageFiles oc3 ic3;

        fprintf('setting up semantic similarities\n');

        s = getContentsOfFile(outFile);
        fprintf('check\n');

        var = 'testVsTestSemanticSimilarities';
        if ~isfield(s,var)
            s.(var) = nan(length(testCategories),length(testCategories));
            s.([var 'f']) = @(is,i,j) pairwiseScore(is,var,testCategories{i},testCategories{j},i,j);
        end
        fprintf('check\n');
        var = 'vocabVsTestSemanticSimilarities';
        if ~isfield(s,var)
            s.(var) = nan(length(vocabCategories),length(testCategories));
            s.([var 'f']) = @(is,i,j) pairwiseScore(is,var,vocabCategories{i},testCategories{j},i,j);
        end
        fprintf('check\n');
        var = 'vocabVsVocabSemanticSimilarities';
        if ~isfield(s,var)
            s.(var) = nan(length(vocabCategories),length(vocabCategories));
            s.([var 'f']) = @(is,i,j) pairwiseScore(is,var,vocabCategories{i},vocabCategories{j},i,j);
        end
        fprintf('check\n');
        var = 'fullVocabVsTestSemanticSimilarities';
        if ~isfield(s,var)
            s.(var) = nan(length(vocabCategories),size(c2,2));
        end
        fprintf('check\n');

        if isnan(s.testVsTestSemanticSimilarities(end,end))
            s = pairwiseScores(s,'testVsTestSemanticSimilarities',outFile,testCategories,testCategories);
            save(outFile,'-v7.3','-append','-struct','s','testVsTestSemanticSimilarities');
        end
        fprintf('check\n');
        if isnan(s.vocabVsTestSemanticSimilarities(end,end))
            s = pairwiseScores(s,'vocabVsTestSemanticSimilarities',outFile,vocabCategories,testCategories);
            save(outFile,'-v7.3','-append','-struct','s','vocabVsTestSemanticSimilarities');
        end
        fprintf('check\n');
        if isnan(s.fullVocabVsTestSemanticSimilarities(end,end))
            for i = 1:length(vocabCategories)
                for j = 1:size(c2,2)
                    s.fullVocabVsTestSemanticSimilarities(i,j) = ...
                      s.vocabVsTestSemanticSimilarities(i,ceil(j/p.nImgs));
                end
            end
            save(outFile,'-v7.3','-append','-struct','s','fullVocabVsTestSemanticSimilarities');
        end
        fprintf('check\n');
        system(['touch ' outfile '.flag']);
    end
end

function s = pairwiseScores(s,var,outFile,wnids1,wnids2)
    simFile = '/data/josh/ruleRiesenhuber2013/src/external/rwSimilarity.pl';
    simDir = '/data/josh/ruleRiesenhuber2013/imageSets/imageNet/semanticSimilarities/';
    [idxIs,idxJs] = meshgrid(1:length(wnids1),1:length(wnids2));
    parfor i1 = 1:numel(idxIs)
        if ~exist([simDir wnids1{idxIs(i1)} '.' wnids2{idxJs(i1)} '.similarity'],'file')
            perl(simFile,wnids1{idxIs(i1)},wnids2{idxJs(i1)},simDir);
        end
    end
    fprintf('-- finished %d remaining\n',numel(idxIs));
    if ~isfield(s,var)
        fprintf('collating pairwise scores\n');
        for i1 = 1:length(wnids1)
            tmp = nan(length(wnids2),1);
            for i2 = 1:length(wnids2)
                data = load([simDir wnids1{i1} '.' wnids2{i2} '.similarity']);
                tmp(i2) = data;
            end
            s.(var)(i1,:) = tmp;
        end
        fprintf('cached %s\n',var);
    end
end

function [s,data] = pairwiseScore(s,var,wnid1,wnid2,i,j)
    simFile = '/data/josh/ruleRiesenhuber2013/src/external/rwSimilarity.pl';
    simDir = '/data/josh/ruleRiesenhuber2013/imageSets/imageNet/semanticSimilarities/';
    if ~exist([simDir wnid1 '.' wnid2 '.similarity'],'file')
        perl(simFile,wnid1,wnid2,simDir);
    end
    data = load([simDir wnid1 '.' wnid2 '.similarity']);
    s.(var)(i,j) = data;
end
