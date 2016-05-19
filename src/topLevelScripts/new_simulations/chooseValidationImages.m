function validation = chooseValidationImages(valDir)
% validation = chooseValidationImages(valDir)
%
% setup the ILSVRC2015 validation set
%
% valDir: string, the directory containing the validation set 
%
% validation: table, {filenames, synsets, validation vs. training}

    validation = table;
    % add the training images
    catDir = [valDir 'train/'];
    cats = dir(catDir);
    cats = {cats.name};
    fprintf('0');
    for iCat = 1:length(cats)
        if (mod(iCat,100) == 0), fprintf(', %d',iCat); end;
        % list all the images in the category
        catInfo = dir([catDir cats{iCat} '/*.JPEG']);
        nImgs = length(catInfo);

        catImgs = strcat(catDir,cats{iCat},'/',{catInfo.name}');

        type = repmat({'training'},nImgs,1);

        thisCat = repmat({cats{iCat}},nImgs,1);

        tmpTable = table(catImgs,thisCat,categorical(type), ...
          'VariableNames',{'file','synset','type'});
        validation = [validation;tmpTable];
    end
    fprintf('\n');

    % add the validation images
    valValInfo = dir([valDir 'val/*.JPEG']);
    valImgs = strcat(valDir,'val/',{valValInfo.name}');
    type = repmat({'validation'},length(valImgs),1);
    thisCat = repmat({'NA'},length(valImgs),1);
    tmpTable = table(valImgs,thisCat,categorical(type), ...
      'VariableNames',{'file','synset','type'});
    validation = [validation;tmpTable];
end
