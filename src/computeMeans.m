function computeMeans(file)
% computeMeans(file)
%
% Compute the mean pixel value for a set of images and write it to disk.
%
% Args:
% - file: string, filename of image files to summarize with mean pixel
    % Get the list of image filenames.
    tab = readtable(file,'Delimiter','space','ReadVariableNames',false);
    files = tab{:,1};

    % Initialize lists of per-image means and the weight of each mean.
    rMeans = [];
    rLens = [];
    gMeans = [];
    gLens = [];
    bMeans = [];
    bLens = [];

    % For each image file
    for iFile = 1:length(files)
        % Read in the image, correcting its colorspace if necessary.
        try
            img = imread(files{iFile});
        catch
            system(['convert ' files{iFile} ' -colorspace rgb ' files{iFile}]);
            img = imread(files{iFile});
        end

        % Compute the mean pixel and its weight.
        rs = img(:,:,1);
        rMeans(end+1) = mean(rs(:));
        rLens(end+1) = numel(rs);

        if size(img,3) == 3
            gs = img(:,:,2);
            bs  = img(:,:,3);
            gMeans(end+1) = mean(gs(:));
            gLens(end+1) = numel(gs);
            bMeans(end+1) = mean(bs(:));
            bLens(end+1) = numel(bs);
        end
    end

    % Compute the final mean pixel.
    r = uint8(sum(rMeans.*(rLens./sum(rLens))));
    g = uint8(sum(gMeans.*(gLens./sum(gLens))));
    b = uint8(sum(bMeans.*(bLens./sum(bLens))));

    % create and save the output
    [d,f,e] = fileparts(file);
    outtable = table(r,g,b,'VariableNames',{'Red','Green','Blue'});
    writetable(outtable,[d '/' f '_means.csv']);
end
