function outFile = cacheVisualSimilarities(outDir,vocabTable,testTable,basetype)
    % make sure we have somewhere to write the results 
    ensureDir(outDir);

    % limit ourselves to just the training images, both for vocab and test
    vocabTable = vocabTable(strcmp(vocabTable.type,'training'),:);
    testTable = testTable(strcmp(testTable.type,'training'),:);

    % compute the vocabulary categories 
    % note: sorting puts them in the correct order
    categories = unique(vocabTable.synset);

    % convert the synsets to be categorical
    vocabTable.synset = categorical(vocabTable.synset);

    % create and load the count file
    count = getCurrentCount([outDir basetype]);

    outFile = [outDir basetype '_visual_similarities.mat'];
    if count < length(categories) % if we haven't finished the task
        % load all the test images
        testFeatures = load_them(basetype, testTable.file)';

        % load any existing results
        if exist(outFile, 'file')
            load(outFile,'-mat','scores');
        else
            scores = nan(height(testTable),numel(categories));
        end

        for iCat = (count+1):length(categories) % for each category
            % load the vocab images for this category
            idxs = find(vocabTable.synset==categories{iCat});
            vf = load_them(basetype, vocabTable.file(idxs))';

            % compute the mean correlation for the category for each test image
            scores(:,iCat) = mean(corr(testFeatures, vf), 2);

            % save the results every so often
            if mod(iCat,20) == 0
                fprintf('%d/%d %f\n',iCat,length(categories),posixtime(datetime));
            end
            if mod(iCat,200) == 0
                save(outFile,'-mat','-v7.3','scores');
                countFile = [outDir basetype '_visual_similarities_count.mat'];
                save(countFile,'count');
            end

            count = iCat;
        end

    else % otherwise, just load the file and return the results
        fprintf('matrix found!\n');
    end
end

function count = getCurrentCount(outStem)
    countFile = [outStem '_visual_similarities_count.mat'];
    if ~exist(countFile,'file')
        count = 0;
        save(countFile,'count');
        fprintf('count file created\n');
    end
    load(countFile,'count');
end

function res = load_it(basetype, filename)
    [d,f,~] = fileparts(filename);
    imgFile = [d '/' f '.' basetype '_gen_mat'];
    load(imgFile,'-mat','c2');
    res = reshape(c2,1,[]);
end

function features = load_them(basetype, filenames)
    first_features = load_it(basetype, filenames{1});
    nFeatures = numel(first_features);
    features = nan(length(filenames), nFeatures);
    features(1,:) = first_features;
    for i = 2:length(filenames)
        features(i,:) = load_it(basetype, filenames{i});
        if (mod(i,25000) == 0 || ismember(i,[1 2 5 10 20 50 100 200 500 1000 2000 5000 10000])) && length(filenames) > 10000
            fprintf('%d/%d %f\n',i,length(filenames),posixtime(datetime));
        end
    end
end
