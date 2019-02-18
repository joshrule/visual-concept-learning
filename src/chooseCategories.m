function [featureCats,evalCats] = ...
    chooseCategories(imgNetDir,allCategoryFile,evalFile,nImgs,nCategories,user,key,srFile)

    % which categories will be our test categories?
    toEvaluate = readtable(evalFile,'Delimiter','comma');
    evalCats = toEvaluate.synset;

    % do we have enough vocabulary concept categories?
    prepareImageNetArchives(imgNetDir)
    downloaded = listDownloadedCategories(imgNetDir);
    available = setdiff(downloaded.synset,evalCats);
    enough = downloaded.synset(downloaded.count >= nImgs);
    qualified = intersect(available,enough);

    % if not, let's pick a few
    allCategories = readtable(allCategoryFile,'Delimiter','comma');
    nRemaining = nCategories-length(qualified);
    if nRemaining > 0
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

        % download the ones we've chosen
        downloadImageNetSynsets(newlyChosen,user,key,srFile,imgNetDir);

        % pick a full set of nCategories for evaluation
        prepareImageNetArchives(imgNetDir);
        downloaded = listDownloadedCategories(imgNetDir);
        available = setdiff(downloaded.synset,evalCats);
        enough = downloaded.synset(downloaded.count >= nImgs);
        qualified = intersect(available,enough);
    end
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
