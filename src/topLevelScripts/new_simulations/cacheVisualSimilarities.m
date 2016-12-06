function visualSimilarities = cacheVisualSimilarities(outDir,vocabTable,testTable,basetype)
    ensureDir(outDir);

    % limit ourselves to just the training images
    vocabTable = vocabTable(strcmp(vocabTable.type,'training'),:);
    testTable = testTable(strcmp(testTable.type,'training'),:);

    % compute the categories in the first list
    categories = unique(vocabTable.synset);

    outFile = [outDir 'visual_similarities.mat'];
    if ~exist(outFile,'file')
        for iRow = 1:height(testTable)
            % compute file name
            [d,f,~] = fileparts(testTable.file(iRow));
            imgFile = [d '/' f '.' basetype '_gen_mat'];
            % load the image's features
            load(imgFile,'-mat','c2');
            srcC2 = reshape(c2,[],1); clear c2;

            for iCat = 1:length(categories)
                % compute images in category
                corrFile = [outDir categories{iCat} '_' f '_visual.mat'];
                if ~exist(corrFile,'file')
                    vocabImgs = vocabTable.file(strcmp(vocabTable.synset,categories{iCat}));
                    img_corr = nan(length(vocabImgs),1);
                    parfor iImg = 1:length(vocabImgs)
                        img_corr(iImg) = score_images(srcC2,vocabImgs(iImg),basetype);
                    end
                    % compute the mean correlation for this category
                    data = mean(img_corr);
                    % save that mean correlation
                    save([outDir categories{iCat} '_' f '_visual.mat'],'-mat','data');
                else
                    load(corrFile,'-mat','data');
                end
                visualSimilarities(iCat,iRow) = data;
            end
        end
        save(outFile,'-mat','-v7.3','visualSimilarities');
        fprintf('matrix saved!\n');
    else
        load(outFile,'-mat','visualSimilarities');
        fprintf('matrix loaded!\n');
    end
end

function correlation = score_images(srcC2,imgName,basetype)
    % compute file name
    [d,f,~] = fileparts(imgName);
    imgFile = [d '/' f '.' basetype '_gen_mat'];
    % load the image's features
    load(imgFile,'-mat','c2');
    vocabC2 = reshape(c2,[],1); clear c2;
    % compute and store the correlation between features
    correlation = corr(srcC2,vocabC2);
end
