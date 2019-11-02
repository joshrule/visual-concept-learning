function [featureCats,evalCats] = ...
    chooseCategories(imgNetDir,allCategoryFile,evalFile,nImgs,nCategories,user,key,srFile)
% [featureCats,evalCats] = chooseCategories(imgNetDir,allCategoryFile,evalFile,nImgs,nCategories,user,key,srFile)
%
% Choose two lists of ImageNet categories: one to serve as vocabulary concepts,
% and one to serve as evaluation concepts. Return both lists.
%
% Args:
% - imgNetDir: string, directory where ImageNet files are stored
% - allCategoryFile: string, filename of a file listing all ImageNet categories
% - evalFile: string, filename of file containing list of evaluation concepts
% - nImgs: int, the minimum number of images a category must contain for use in
%   our simulations
% - nCategories: int, the number of categories to be selected for `featureCats`
% - user: string, an ImageNet username
% - key: string, the corresponding ImageNet password
% - srFile: string, filename of file containing ImageNet structure

    % Which categories will be our test categories?
    % These come from the ILSVRC and so are easier to pick.
    toEvaluate = readtable(evalFile,'Delimiter','comma');
    evalCats = toEvaluate.synset;

    % Do we have enough vocabulary concept categories?
    prepareImageNetArchives(imgNetDir)
    downloaded = listDownloadedCategories(imgNetDir);
    available = setdiff(downloaded.synset,evalCats);
    enough = downloaded.synset(downloaded.count >= nImgs);
    qualified = intersect(available,enough);

    allCategories = readtable(allCategoryFile,'Delimiter','comma');
    nRemaining = nCategories-length(qualified);
    % If not...
    if nRemaining > 0
        % Choose a few, based on user input (to validate concreteness).
        fprintf('\tchoosing %d more categories\n',nRemaining);
        newlyChosen = {};
        while nRemaining > 0
            options = setdiff(allCategories.synset(allCategories.count >= nImgs), ...
                              union(union(evalCats,qualified), ...
                                    newlyChosen));
            newest = options(randperm(length(options), ...
                                      1));

            try 
                fprintf('%d: %s - %s  ',nRemaining,newest{1},wnidToWords(srFile, newest{1}));
                str = input('select (y/N)? ','s');

                if ~isempty(regexpi(str,'y(es)?'))
                    newlyChosen = {newlyChosen{:} newest{:}}';
                    nRemaining = nRemaining-1;
                end
            catch
            end
        end

        % Download the ones we've chosen.
        downloadImageNetSynsets(newlyChosen,user,key,srFile,imgNetDir);

        % Choose a full set of nCategories for evaluation.
        prepareImageNetArchives(imgNetDir);
        downloaded = listDownloadedCategories(imgNetDir);
        available = setdiff(downloaded.synset,evalCats);
        enough = downloaded.synset(downloaded.count >= nImgs);
        qualified = intersect(available,enough);
    end

    % Randomly select an appropriately sized subset to return.
    featureCats = qualified(randperm(size(qualified,1),nCategories));
end

function down = listDownloadedCategories(imgNetDir)
    synsetDirs = dir([imgNetDir 'n*']);
    nImgs = nan(length(synsetDirs),1);
    parfor i = 1:length(synsetDirs)
        nImgs(i) = length(dir([imgNetDir synsetDirs(i).name '/*JPEG']));
    end
    down = table(nImgs,{synsetDirs.name}','VariableNames',{'count','synset'});
end
