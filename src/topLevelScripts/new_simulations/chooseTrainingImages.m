function training = chooseTrainingImages(cats,catDir,N)
% training = chooseTrainingImages(cats,catDir,N)
%
% choose N images/category for validation and the rest for training
%
% cats: cell array of strings, the list of categories
% catDir: string, the directory where these categories are stored
% N: integer, the number of images per category to reserve for validation
%
% training: table, {filename, category, training/validation}

    training = table;
    fprintf('0');
    for iCat = 1:length(cats)
        if (mod(iCat,100) == 0), fprintf(', %d',iCat); end;
        % list all the images in the category
        catInfo = dir([catDir cats{iCat} '/*.JPEG']);
        nImgs = length(catInfo);

        catImgs = strcat(catDir,cats{iCat},'/',{catInfo.name}');

        validation = repmat({'training'},nImgs,1);
        
        % don't need validation images AND test images
%       valAndTestImgs = randperm(nImgs,3*N);
        valAndTestImgs = randperm(nImgs,N);

        valImgs = valAndTestImgs(1:N);
        validation(valImgs) = {'validation'};

        % don't need validation images AND test images
%       testImgs = valAndTestImgs((N+1):end);
%       validation(testImgs) = {'testing'};

        thisCat = repmat({cats{iCat}},nImgs,1);

        tmp = table(catImgs,thisCat,categorical(validation), ...
          'VariableNames',{'file','synset','type'});
        training = [training;tmp];
    end
    fprintf('\n');
end
