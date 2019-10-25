function prepLMDBFiles(caffe_dir,outDir)
    feTrFile = [caffe_dir 'feature_training_images.txt'];
    feVaFile = [caffe_dir 'feature_validation_images.txt'];
    feImgFile = [outDir 'trainingImages.csv'];

    prep_lmdb_helper(feTrFile,feVaFile,feImgFile);

    evTrFile = [caffe_dir 'evaluation_training_images.txt'];
    evVaFile = [caffe_dir 'evaluation_validation_images.txt'];
    evImgFile = [outDir 'validationImages.csv'];

    prep_lmdb_helper(evTrFile,evVaFile,evImgFile);
end

function prep_lmdb_helper(trainFile,valFile,imgFile)
    if ~exist(trainFile,'file') || ~exist(valFile,'file')
        imgs = readtable(imgFile,'Delimiter','comma');
        synsets = unique(imgs.synset);
        trainRows = strcmp(imgs.type,'training');
        valRows   = strcmp(imgs.type,'validation');

        prep_lmdb_helper_helper(imgs,trainRows,synsets,trainFile)
        prep_lmdb_helper_helper(imgs,valRows,synsets,valFile)
    end
end

function prep_lmdb_helper_helper(imgs,rows,synsets,file)
    if ~exist(file,'file')
        [~,labels] = ismember(imgs.synset(rows),synsets);

        tab = table(imgs.file(rows),labels);
        writetable(tab,file,'Delimiter',' ','WriteVariableNames',false);
        fprintf('%s written\n',file);
    else
        fprintf('%s found\n',file);
    end
end
