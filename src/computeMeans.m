function compute_means(file)
    tab = readtable(file,'Delimiter','space','ReadVariableNames',false);
    files = tab{:,1};

    rMeans = [];
    rLens = [];

    gMeans = [];
    gLens = [];

    bMeans = [];
    bLens = [];

    for iFile = 1:length(files)
        try
            img = imread(files{iFile});
        catch
            system(['convert ' files{iFile} ' -colorspace rgb ' files{iFile}]);
            img = imread(files{iFile});
        end

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

    r = uint8(sum(rMeans.*(rLens./sum(rLens))));
    g = uint8(sum(gMeans.*(gLens./sum(gLens))));
    b = uint8(sum(bMeans.*(bLens./sum(bLens))));

    % create and save the output
    [d,f,e] = fileparts(file);
    outtable = table(r,g,b,'VariableNames',{'Red','Green','Blue'});
    writetable(outtable,[d '/' f '_means.csv']);
end
