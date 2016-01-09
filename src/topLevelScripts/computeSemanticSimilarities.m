function computeSemanticSimilarities(p,c2Files,organicC3Files,inorganicC3Files)
% computeSemanticSimilarities(p,c2Files,organicC3Files,inorganicC3Files)
    outFile = [p.outDir 'semantic-similarities.mat'];
    if exist(outFile,'file') 
        load(outFile);
    end

    c2 = buildC2(c2Files);
    [oc3,olabels,oImageFiles] = buildC3(organicC3Files);
    [ic3,ilabels,iImageFiles] = buildC3(inorganicC3Files);
    c3 = [oc3; ic3];
    assert(isequal(ilabels,olabels), 'Failed label equality in semantic analysis 1\n');
    assert(isequal(iImageFiles,oImageFiles), 'Failed image list equality in semantic analysis 1\n');
    clear ilabels olabels oImageFiles iImageFiles oc3 ic3;

    if ~(exist('testCategories','var') && exist('vocabCategories','var'))
        testCategories = listImageNetCategories(c2Files);
        oC3Cats = load([p.outDir 'organic-categories.mat'],'c3Categories');
        iC3Cats = load([p.outDir 'inorganic-categories.mat'],'c3Categories');
        vocabCategories = [reshape(oC3Cats.c3Categories,[],1); reshape(iC3Cats.c3Categories,[],1)];
        size(vocabCategories)
        save(outFile,'testCategories','vocabCategories','-v7.3');
    end

    pairwiseScores(outFile,'testVsTestSimilarities',p.simFile,testCategories,testCategories);
    fprintf('similarity 1 computed\n');
    pairwiseScores(outFile,'vocabVsTestSimilarities',p.simFile,vocabCategories,testCategories);
    fprintf('similarity 2 computed\n');
%   compute this manually and insert into file because this takes too long with only 12 processors
%   pairwiseScores(outFile,'vocabVsVocabSimilarities',p.simFile,vocabCategories,vocabCategories);
%   fprintf('similarity 3 computed\n');
    load(outFile);

    if ~(exist('fullVocabVsTestSimilarities','var'))
        for i = 1:length(vocabCategories)
            for j = 1:size(c2,2)
                fullVocabVsTestSimilarities(i,j) = ...
                  vocabVsTestSimilarities(i,ceil(j/p.nImgs));
            end
        end
        save(outFile,'fullVocabVsTestSimilarities','-v7.3','-append');
    end
    fprintf('similarity 4 computed\n');
end

function pairwiseScores(outFile,var,simFile,wnids1,wnids2)
    s = load(outFile);
    fprintf('loaded the file %s\n',outFile);
    if ~isfield(s,var)
        m = nan(length(wnids1),length(wnids2));
        s = setfield(s,var,m);
        save(outFile,'-struct','s');
        fprintf('created %s\n',var);
    else
        m = getfield(s,var);
        fprintf('loaded %s\n',var);
    end
    [idxIs,idxJs] = ind2sub(size(m),find(isnan(m)));
    for i1 = 1:length(idxIs)
        m(IdxIs(i1),IdxJs(i1)) = str2num(perl(simFile,wnids1{IdxIs(i1)},wnids2{IdxJs(i1)}));
        if (mod(i1,1000)==0)
            fprintf('%d/%d, ',i1,length(idxIs));
            s = setfield(s,var,m);
            save(outFile,'-struct','s');
        end
    end
    fprintf('-- finished %d remaining\n',length(idxIs));
    fprintf('computed %s\n',var);
end
