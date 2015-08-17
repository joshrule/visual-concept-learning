function [categories,imgLists] = chooseCategories(outFile,imgSetDir,catFile,nImgs,restrictedImgs,exts)
        load(catFile,'categories');
        potentialCategories = categories;
        clear categories;
        rngState =rng;
        categories = {};
        imgLists = restrictedImgs;
        for iCat = 1:length(potentialCategories)
            % for each category, choose a minimum number of images that are unique
            category = potentialCategories{iCat};
            catDir = [imgSetDir category '/'];
            [success,newImgs] = chooseNDuplicateFreeImags( ...
                nImgs,restrictedImages,catDir,exts);
            if success > 0
                categories = {categories{:} category};
                imgLists = {imgLists{:} newImgs};
            end
        end
end

function [count,newImgs] = chooseNDuplicateFreeImages(N,restrictedImgs,dir,exts)
% [success,newImgs] = chooseImages(nImgs,restrictedImgs,dir)
%
% create a duplicate free list of specified length given a list of already
% chosen images and a new category/directory of images.
%
% N: double scalar, the number of new (non-duplicate) images to find
% restrictedImages: cell array of cell array of strings, already chosen images
% dir: string, the directory holding new image files
% exts: cell array of strings, the potential image file extensions 
%
% count: double scalar, the number of new images found (max of N)
% imgFilesOut: cell array of strings, 'count' new images 
    allImgs = lsDir(dir,exts);
    allImgs = allImgs(randperm(length(allImgs)));

    count = 0;
    iImg = 1;
    restrictedImgVec = cat(2,restrictedImgs);
    while (count <= nImgs) && (iImg <= length(allImgs))
        currImg = allImgs{iImg};
        currFiles = {restrictedImgVec{:} newImgs{:}};
        [~,output] = system(['ls -1' strjoin(currFiles,' ') ...
            ' | xargs -P 32 -n 4096 diff -s --from-file=' ...
            currImg ' | grep identical']);
        notDuplicate = isempty(strfind(output,'identical'));
        if notDuplicate
            count = count+1;
            newImgs{count} = currImg;
        end
        iImg = iImg+1;
    end
    if count < nImgs
        imgFilesOut = [];
        count = -1;
    end
    newImgs = newImgs';
end

function files = lsDir(dir,exts)
    files = {};
    for ext = exts
        newFiles = dir([dir '*.' ext]);
        files = {files{:} strcat(dir, {newFiles.name}')};
    end
end
