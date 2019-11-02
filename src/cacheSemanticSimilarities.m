function outFile = cacheSemanticSimilarities(outDir,conceptual,imageTable)
% outFile = cacheSemanticSimilarities(outDir,conceptual,imageTable)
%
% Save semantic similarities between categories using Wu-Palmer Distance in
% WordNet.
%
% Args:
% - outDir: string, directory to which to write the similarities.
% - conceptual: cell vector, a list of the concepts/vocabulary categories
% - imageTable: table, a table listing images, their type (training. vs.
%   validation), and their category/synset.
    ensureDir(outDir);
    imageTable = imageTable(strcmp(imageTable.type,'training'),:);
    outFile = cacheThesePairs(outDir,conceptual,imageTable);
end

function outFile = cacheThesePairs(outDir,vocabulary,imageTable)
    % in the original experiments, this was 2,000 x 1000 categories = 2,000,000 computations
    outFile = [outDir 'semantic_similarities.mat'];
    if ~exist(outFile,'file')

        testConcepts = unique(imageTable.synset);

        scores = nan(length(vocabulary),length(testConcepts));
        for i1 = 1:length(vocabulary)
            parfor i2 = 1:length(testConcepts)
                scores(i1,i2) = cachePair(outDir,vocabulary{i1},testConcepts{i2});
            end
        end
        fprintf('caching complete!\n');

        scores = nan(length(vocabulary),height(imageTable));
        for i1 = 1:length(vocabulary)
            for i2 = 1:length(testConcepts)
                theseImages = find(strcmp(imageTable.synset,testConcepts{i2}));
                for i3 = 1:length(theseImages)
                    i3idx = theseImages(i3);
                    scores(i1,i3idx) = scores(i1,i2);
                end
            end
        end
        fprintf('master matrix built!\n');
        save(outFile,'-mat','-v7.3','scores');
        fprintf('matrix saved!\n');
    else
        fprintf('matrix found!\n');
    end
end

function score = cachePair(outDir,vocabword,testword)
    thePair = sort({vocabword,testword});
    file_name = [outDir thePair{1} '_' thePair{2} '_semantic.mat'];
    if ~exist(file_name,'file')
        score = pairwiseScore(file_name,thePair{1},thePair{2});
    else
        load(file_name,'-mat','data');
        score = data;
    end
end

function data = pairwiseScore(file,wnid1,wnid2)
    simFile = '/data1/josh/ruleRiesenhuber2013/src/utils/metrics/similarity.pl';
    simDir  = '/data1/josh/ruleRiesenhuber2013/tmp/';
    ensureDir(simDir);
    perl(simFile,wnid1,wnid2,simDir);
    data = load([simDir wnid1 '.' wnid2 '.similarity']);
    save(file,'data');
    delete([simDir wnid1 '.' wnid2 '.similarity']);
end
