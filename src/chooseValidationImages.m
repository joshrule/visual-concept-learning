function tab = chooseValidationImages(valDir)
% tab = chooseValidationImages(valDir)
%
% setup the ILSVRC2015 validation set
%
% valDir: string, the directory containing the validation set 
%
% tab: table, {filenames, synsets, validation vs. training}

    cat_dir = [valDir 'Data/CLS-LOC/train/'];
    training = add_from_tree(cat_dir,'training');

    img_dir = dir([valDir 'Data/CLS-LOC/val/']);
    annote_dir = [valDir 'Annotations/CLS-LOC/val/'];
    validation = add_from_one_dir(img_dir,annote_dir,'validation');

    % NOT reading the test images, as these aren't labeled

    tab = [training;validation];
end

function tab = add_from_one_dir(img_dir,annote_dir,label)
    info = dir([img_dir '*.JPEG']);
    
    imgs = strcat(img_dir,{info.name}');

    type = repmat({label},length(valImgs),1);

    count = 0;
    for iImg = 1:length(imgs)
        [~,f,~] = fileparts(imgs{iImg});
    
        xml = parsexml([annote_dir f '.xml']);
        objs = xml.Children(strcmp({xml.Children.Name},'object'));
        cats = {};
        ns = {};
        for i = 1:length(objs)
            ns{i} = objs(i).Children(strcmp({objs(i).Children.Name},'name')).Children.Data;
        end;
        uns = unique(ns);
        for i = 1:length(uns)
            count = count + 1;
            tableImgs{count} = valImgs{iImg};
            tableSynsets{count} = uns{i};
        end
    end

    tab = table(tableImgs,tableSynsets',categorical(type), ...
      'VariableNames',{'file','synset','type'});
    fprintf('finished with %s\n',label);
end

function tab = add_from_tree(cat_dir,label)
    cats = dir(cat_dir);
    cats = {cats.name};
    fprintf('0');
    tab = table;
    for iCat = 1:length(cats)
        if (mod(iCat,100) == 0), fprintf(', %d',iCat); end;
        % list all the images in the category
        catInfo = dir([cat_dir cats{iCat} '/*.JPEG']);
        nImgs = length(catInfo);

        catImgs = strcat(cat_dir,cats{iCat},'/',{catInfo.name}');

        type = repmat({label},nImgs,1);

        thisCat = repmat({cats{iCat}},nImgs,1);

        tmp = table(catImgs,thisCat,categorical(type), ...
          'VariableNames',{'file','synset','type'});
        tab = [tab;tmp];
    end
    fprintf('\n');
    fprintf('finished with %s\n',label);
end
