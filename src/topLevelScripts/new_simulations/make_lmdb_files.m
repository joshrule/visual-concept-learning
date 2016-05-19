% feature images
if ~exist('../../../caffe/feature_training_images.txt','file') || ...
   ~exist('../../../caffe/feature_validation_images.txt','file')
    trImgs = readtable('../../../evaluation/v0_2/trainingImages.csv');

    synsets = unique(trImgs.synset);

    trTrRows = strcmp(trImgs.type,'training');
    trVaRows = strcmp(trImgs.type,'validation');

    [~,trTrLabels] = ismember(trImgs.synset(trTrRows),synsets);

    trTrImgs = table(trImgs.file(trTrRows),trTrLabels);
    writetable(trTrImgs,'../../../caffe/feature_training_images.txt', ...
      'Delimiter',' ','WriteVariableNames',false);

    [~,trVaLabels] = ismember(trImgs.synset(trVaRows),synsets);

    trVaImgs = table(trImgs.file(trVaRows),trVaLabels);
    writetable(trVaImgs,'../../../caffe/feature_validation_images.txt', ...
      'Delimiter',' ','WriteVariableNames',false);
end

% evaluation images
if ~exist('../../../caffe/evaluation_training_images.txt','file') || ...
   ~exist('../../../caffe/evaluation_validation_images.txt','file')
    vaImgs = readtable('../../../evaluation/v0_2/validationImages.csv');

    % NA isn't actually a synset
    synsets = setdiff(unique(vaImgs.synset),{'NA'});

    vaTrRows = strcmp(vaImgs.type,'training');
    vaVaRows = strcmp(vaImgs.type,'validation');

    % training images
    if ~exist('../../../caffe/evaluation_training_images.txt','file')
        [~,vaTrLabels] = ismember(vaImgs.synset(vaTrRows),synsets);

        vaTrImgs = table(vaImgs.file(vaTrRows),vaTrLabels);
        writetable(vaTrImgs,'../../../caffe/evaluation_training_images.txt', ...
          'Delimiter',' ','WriteVariableNames',false);
        fprintf('evaluation_training_images.txt written\n');
    else
        fprintf('evaluation_training_images.txt found\n');
    end

    % validation images
    if ~exist('../../../caffe/evaluation_validation_images.txt','file')
        vaVaImages = vaImgs.file(vaVaRows);
        tableImgs = {};
        tableSynsets = {};
        annote_dir = '/data2/image_sets/image_net/ILSVRC2015/Annotations/CLS-LOC/val/';

        count = 0;
        fprintf('%d',length(vaVaImages));
        for iImg = 1:length(vaVaImages)
            if mod(iImg,500) == 0
                fprintf('%d',iImg);
            elseif mod(iImg,100) == 0
                fprintf('.');
            end
            [~,f,~] = fileparts(vaVaImages{iImg});
        
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
                tableImgs{count} = vaVaImages{iImg};
                tableSynsets{count} = uns{i};
            end
        end
        fprintf('\n');
     
        [~,tableLabels] = ismember(tableSynsets,synsets);

        vaVaImgs = table(tableImgs',tableLabels');
        writetable(vaVaImgs,'../../../caffe/evaluation_validation_images.txt', ...
          'Delimiter',' ','WriteVariableNames',false);
        fprintf('evaluation_validation_images.txt written\n');
    else
        fprintf('evaluation_validation_images.txt found\n');
    end
end

% cleanup
clear
