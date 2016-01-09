% The point of this analysis is that we want some way to quantitatively explain
% the information in C3 features. I can tell you that C3-space provides a
% description of goodness-of-fit between an image and several semantic
% categories based on visual similarity, but that is qualitative. It would be
% great to more quantitatively show how similarity in C3-space is affected by
% visual similarity or semantic similarity. We want some explanation of what
% information is captured in C3-space.

% We could look at similarity in C3-space in two ways. First, we could define
% similarity across the entire space. How does the visual and semantic
% similarity between two test categories predict their C3-space similarity
% across all 2,000 C3 units? Or, how does the C3-space similarity vary between
% images in a category? Our 2,000 categories were arbitrarily selected,
% however, so there's no reason to trust this particular set of features. We
% may instead want to figure out how individual units relate to each other. On
% this second option, we could define similarity with respect to a single C3
% unit. We'd look at how two different test categories are treated by the same
% C3 unit as well as how the same test category will differ across C3 units. 

% Let's run both types of analysis. They should be simple to implement and
% quick to run.

function semanticAnalysis(p,c2Files,organicC3Files,inorganicC3Files)
    fPer = [p.outDir 'kmeans-combined-evaluation.mat'];
    fC3 = [p.outDir 'c3-similarities.mat'];
    fVis = [p.outDir 'visual-similarities.mat'];
    fSem = [p.outDir 'semantic-similarities.mat'];
    [~,~,c2Images] = buildC2(c2Files);
    c3CategoriesO = load([p.outDir 'organic-categories.mat'],'c3Categories');
    c3CategoriesI = load([p.outDir 'inorganic-categories.mat'],'c3Categories');
    c3Categories = [c3CategoriesO.c3Categories; c3CategoriesI.c3Categories];

    analysisOver1C2AndAllC3(p,{c2Images,c2Images},fC3,fVis,fSem);
    analysisOver2C2And1C3(p,{c2Images,c2Images,c3Categories},fPer,fVis,fSem);
    analysisOver1C2And2C3(p,{c2Images,c3Categories,c3Categories},fPer,fVis,fSem);
end

% For the first analysis, predicting C3 similarity based on visual and semantic
% similarity, we do so as follows: for each pair of test images, compute the
% C3, visual, and semantic similarity. Then, use regression (a GLM perhaps) to
% predict the C3 similarity based on the visual and semantic similarities. So,
% we need to regress either over three 15000x15000 matrices, or use a randomly
% selected sub-sample of some 10,000 pairs of test images. Let's say 10,000
% pairs. So, the basic steps of this analysis:

function analysisOver1C2AndAllC3(p,tupleSpace,fileC3,fileVis,fileSem)
    outFile = [p.outDir 'analysis-over-1-C2-and-All-C3.mat'];
    s = getContentsOfFile(outFile);
    vars = fieldnames(s);
    if ~ismember('pairs',vars);
        fprintf('selecting pairs for 1 vs All\n');
        pairs = selectNRandomTuples(p.nPairs,tupleSpace);
%       pairs = selectNRandomTuples(100,tupleSpace);

        fprintf('loading necessary matrices\n');
        nC3 = 'testVsTestC3Similarity';
        tmp = load(fileC3,nC3,[nC3 'f']);
        vC3 = tmp.(nC3);
        fC3 = tmp.([nC3 'f']);

        nVis = 'testVsTestVisualSimilarities';
        tmp = load(fileVis,nVis,[nVis 'f']);
        vVis = tmp.(nVis);
        fVis = tmp.([nVis 'f']);

        nSem = 'testVsTestSemanticSimilarities';
        tmp = load(fileSem,nSem,[nSem 'f']);
        vSem = tmp.(nSem);
        fSem = tmp.([nSem 'f']);

        for iPair = 1:length(pairs)
            if mod(iPair,100) == 0, fprintf('%d ',iPair); end;
            pairs(iPair).categories = ceil(pairs(iPair).idx ./ p.nImgs);
            [pairs(iPair).c3Similarity,vC3] = getCachedMatrixEntry(vC3,fC3,nC3,pairs(iPair).idx(1),pairs(iPair).idx(2));
            [pairs(iPair).visualSimilarity,vVis] = getCachedMatrixEntry(vVis,fVis,nVis,pairs(iPair).idx(1),pairs(iPair).idx(2));
            [pairs(iPair).semanticSimilarity,vSem] = getCachedMatrixEntry(vSem,fSem,nSem,pairs(iPair).categories(1),pairs(iPair).categories(2));
        end
        fprintf('\nsaving results\n');
        s.pairs = pairs;
        save(outFile,'-v7.3','-struct','s');
        testVsTestC3Similarity = vC3;
        save(fileC3,'-v7.3','-append','testVsTestC3Similarity');
        testVsTestVisualSimilarities = vVis;
        save(fileVis,'-v7.3','-append','testVsTestVisualSimilarities');
        testVsTestSemanticSimilarities = vSem;
        save(fileSem,'-v7.3','-append','testVsTestSemanticSimilarities');
    end
    if ~all(ismember({'betas','deviance','stats'},vars));
        fprintf('fitting model\n');
        [s.betas,s.deviance,s.stats] = ...
          glmfit([[s.pairs.visualSimilarity]' [s.pairs.semanticSimilarity]'], ...
                 [s.pairs.c3Similarity]', ...
                 'normal', ...
                 'link','identity', ...
                 'constant','on');
        save(outFile,'-v7.3','-struct','s');
    end
    fprintf('1 vs all analysis complete\n');
end

% For the second set of analyses, predicting differences in C3 activations
% without categories or within categories, we can take a similar approach.
% First, we'll select a sub-sample of (test, test, C3) triples, compute the
% various pair-wise relations, and try to predict the differences between the
% C3 similarities based on the other similarities. Then, we'll select a
% sub-sample of (test, C3, C3) triples and repeat the process. 

function analysisOver2C2And1C3(p,tupleSpace,fileC3,fileVis,fileSem)
    outFile = [p.outDir 'analysis-over-2-C2-and-1-C3.mat'];
    s = getContentsOfFile(outFile);
    vars = fieldnames(s);
    if ~ismember('pairs',vars);
        fprintf('selecting pairs for 2 vs 1\n');
        pairs = selectNRandomTuples(p.nPairs,tupleSpace);
%       pairs = selectNRandomTuples(100,tupleSpace);

        fprintf('loading necessary matrices\n');
        var = 'm';
        tmp = load(fileC3,var);
        vC3 = tmp.(var);

        nVis1 = 'testVsTestVisualSimilarities';
        nVis2 = 'vocabVsTestVisualSimilarities';
        tmp = load(fileVis,nVis1,[nVis1 'f'],nVis2,[nVis2 'f']);
        vVis1 = tmp.(nVis1);
        fVis1 = tmp.([nVis1 'f']);
        vVis2 = tmp.(nVis2);
        fVis2 = tmp.([nVis2 'f']);

        nSem1 = 'testVsTestSemanticSimilarities';
        nSem2 = 'vocabVsTestSemanticSimilarities';
        tmp = load(fileSem,nSem1,[nSem1 'f'],nSem2,[nSem2 'f']);
        vSem1 = tmp.(nSem1);
        fSem1 = tmp.([nSem1 'f']);
        vSem2 = tmp.(nSem2);
        fSem2 = tmp.([nSem2 'f']);

        for iPair = 1:length(pairs)
            if mod(iPair,100) == 0, fprintf('%d ',iPair); end;
            pairs(iPair).categories = [ceil(pairs(iPair).idx(1:2) ./ p.nImgs) pairs(iPair).idx(3)];
            pairs(iPair).c3Diff = vC3(pairs(iPair).idx(3),pairs(iPair).idx(1))-vC3(pairs(iPair).idx(3),pairs(iPair).idx(2));
            [pairs(iPair).visualSimilarity12,vVis1] = getCachedMatrixEntry(vVis1,fVis1,nVis1,pairs(iPair).idx(1),pairs(iPair).idx(2));
            [x,vVis2] = getCachedMatrixEntry(vVis2,fVis2,nVis2,((pairs(iPair).idx(3)-1)*p.nImgs+1):(pairs(iPair).idx(3)*p.nImgs),pairs(iPair).idx(1));
            pairs(iPair).visualSimilarity13 = mean(mean(x));
            [x,vVis2] = getCachedMatrixEntry(vVis2,fVis2,nVis2,((pairs(iPair).idx(3)-1)*p.nImgs+1):(pairs(iPair).idx(3)*p.nImgs),pairs(iPair).idx(2));
            pairs(iPair).visualSimilarity23 = mean(mean(x));
            [pairs(iPair).semanticSimilarity12,vSem1] = getCachedMatrixEntry(vSem1,fSem1,nSem1,pairs(iPair).categories(1),pairs(iPair).categories(2));
            [pairs(iPair).semanticSimilarity13,vSem2] = getCachedMatrixEntry(vSem2,fSem2,nSem2,pairs(iPair).categories(3),pairs(iPair).categories(1));
            [pairs(iPair).semanticSimilarity23,vSem2] = getCachedMatrixEntry(vSem2,fSem2,nSem2,pairs(iPair).categories(3),pairs(iPair).categories(2));
        end
        fprintf('\nsaving results\n');
        s.pairs = pairs;
        save(outFile,'-v7.3','-struct','s');

        testVsTestVisualSimilarities = vVis1;
        vocabVsTestVisualSimilarities = vVis2;
        save(fileVis,'-v7.3','-append','testVsTestVisualSimilarities','vocabVsTestVisualSimilarities');

        testVsTestSemanticSimilarities = vSem1;
        vocabVsTestSemanticSimilarities = vSem2;
        save(fileSem,'-v7.3','-append','testVsTestSemanticSimilarities','vocabVsTestSemanticSimilarities');
    end
    if ~all(ismember({'betas','deviance','stats'},vars));
        fprintf('fitting model\n');
        [s.betas,s.deviance,s.stats] = ...
          glmfit([[pairs.visualSimilarity12]'   [pairs.visualSimilarity13]'   [pairs.visualSimilarity23]' ...
                  [pairs.semanticSimilarity12]' [pairs.semanticSimilarity13]' [pairs.semanticSimilarity23]'], ...
                 [pairs.c3Diff]', ...
                 'normal', ...
                 'link','identity', ...
                 'constant','on');
        save(outFile,'-v7.3','-struct','s');
    end
    fprintf('2 vs 1 analysis complete\n');
end

function analysisOver1C2And2C3(p,tupleSpace,fileC3,fileVis,fileSem)
    outFile = [p.outDir 'analysis-over-1-C2-and-2-C3.mat'];
    s = getContentsOfFile(outFile);
    vars = fieldnames(s);
    if ~ismember('pairs',vars);
        fprintf('selecting pairs for 1 vs 2\n');
        pairs = selectNRandomTuples(p.nPairs,tupleSpace);
%       pairs = selectNRandomTuples(10,tupleSpace);

        fprintf('loading necessary matrices\n');
        var = 'm';
        tmp = load(fileC3,var);
        vC3 = tmp.(var);

        nVis1 = 'vocabVsTestVisualSimilarities';
        nVis2 = 'vocabVsVocabVisualSimilarities';
        tmp = load(fileVis,nVis1,[nVis1 'f'],nVis2,[nVis2 'f']);
        vVis1 = tmp.(nVis1);
        fVis1 = tmp.([nVis1 'f']);
        vVis2 = tmp.(nVis2);
        fVis2 = tmp.([nVis2 'f']);

        nSem1 = 'vocabVsTestSemanticSimilarities';
        nSem2 = 'vocabVsVocabSemanticSimilarities';
        tmp = load(fileSem,nSem1,[nSem1 'f'],nSem2,[nSem2 'f']);
        vSem1 = tmp.(nSem1);
        fSem1 = tmp.([nSem1 'f']);
        vSem2 = tmp.(nSem2);
        fSem2 = tmp.([nSem2 'f']);

        for iPair = 1:length(pairs)
            if mod(iPair,100) == 0, fprintf('%d ',iPair); end;
            pairs(iPair).categories = [ceil(pairs(iPair).idx(1)/p.nImgs) pairs(iPair).idx(2) pairs(iPair).idx(3)];
            pairs(iPair).c3Diff = vC3(pairs(iPair).idx(2),pairs(iPair).idx(1))-vC3(pairs(iPair).idx(3),pairs(iPair).idx(1));

            fprintf('%d: %d %d %d -- %d %d %d\n',iPair,pairs(iPair).idx,pairs(iPair).categories);
            [x,vVis1] = getCachedMatrixEntry(vVis1,fVis1,nVis1,((pairs(iPair).idx(2)-1)*p.nImgs+1):(pairs(iPair).idx(2)*p.nImgs),pairs(iPair).idx(1));
            pairs(iPair).visualSimilarity12 = mean(x);
            [x,vVis1] = getCachedMatrixEntry(vVis1,fVis1,nVis1,((pairs(iPair).idx(3)-1)*p.nImgs+1):(pairs(iPair).idx(3)*p.nImgs),pairs(iPair).idx(1));
            pairs(iPair).visualSimilarity13 = mean(x);
            [pairs(iPair).visualSimilarity23,vVis2] = getCachedMatrixEntry(vVis2,fVis2,nVis2,pairs(iPair).categories(2),pairs(iPair).categories(3));
            [pairs(iPair).semanticSimilarity12,vSem1] = getCachedMatrixEntry(vSem1,fSem1,nSem1,pairs(iPair).categories(2),pairs(iPair).categories(1));
            [pairs(iPair).semanticSimilarity13,vSem1] = getCachedMatrixEntry(vSem1,fSem1,nSem1,pairs(iPair).categories(3),pairs(iPair).categories(1));
            [pairs(iPair).semanticSimilarity23,vSem2] = getCachedMatrixEntry(vSem2,fSem2,nSem2,pairs(iPair).categories(2),pairs(iPair).categories(3));
        end
        fprintf('\nsaving results\n');
        s.pairs = pairs;
        save(outFile,'-v7.3','-struct','s');
        vocabVsTestVisualSimilarities = vVis1;
        vocabVsVocabVisualSimilarities = vVis2;
        save(fileVis,'-v7.3','-append','vocabVsVocabVisualSimilarities','vocabVsTestVisualSimilarities');
        vocabVsTestSemanticSimilarities = vSem1;
        vocabVsVocabSemanticSimilarities = vSem2;
        save(fileSem,'-v7.3','-append','vocabVsVocabSemanticSimilarities','vocabVsTestSemanticSimilarities');
    end
    if ~all(ismember({'betas','deviance','stats'},vars));
        fprintf('fitting model\n');
        [s.betas,s.deviance,s.stats] = ...
          glmfit([[s.pairs.visualSimilarity12]'   [s.pairs.visualSimilarity13]'   [s.pairs.visualSimilarity23]' ...
                  [s.pairs.semanticSimilarity12]' [s.pairs.semanticSimilarity13]' [s.pairs.semanticSimilarity23]'], ...
                 [s.pairs.c3Diff]', ...
                 'normal', ...
                 'link','identity', ...
                 'constant','on');
        save(outFile,'-v7.3','-struct','s');
    end
    fprintf('1 vs 2 analysis complete\n');
end
