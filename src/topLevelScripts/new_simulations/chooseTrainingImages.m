function training = chooseTrainingImages(cats,catDir,N)
% training = chooseTrainingImages(cats,catDir,N)
%
% choose N images per category for validation and leave the rest for training
%
% cats: cell array of strings, the list of categories
% catDir: string, the directory where these categories are stored
% N: integer, the number of images per category to reserve for validation
%
% training: table, {filename, category, training or validation image}

    training = table;
    fprintf('0');
    for iCat = 1:length(cats)
        if (mod(iCat,100) == 0), fprintf(', %d',iCat); end;
        % list all the images in the category
        catInfo = dir([catDir cats{iCat} '/*.JPEG']);
        nImgs = length(catInfo);

        catImgs = strcat(catDir,cats{iCat},'/',{catInfo.name}');

        valImgs = randperm(nImgs,N);
        validation = repmat({'training'},nImgs,1);
        validation(valImgs) = {'validation'};

        thisCat = repmat({cats{iCat}},nImgs,1);

        tmpTable = table(catImgs,thisCat,categorical(validation), ...
          'VariableNames',{'file','synset','type'});
        training = [training;tmpTable];
    end
    fprintf('\n');
end
