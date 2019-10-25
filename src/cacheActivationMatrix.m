function outFile = cacheActivationMatrix(files,labels,outFile)
% outFile = cacheActivationMatrix(files,labels,outFile)
%
% given a list of files storing activations and their labels,
% create a matrix of the activations *and* a matrix of one-hot 
% vectors encoding the labels and save it to a file.
%
% Args:
% - files: nImgs cell vector: the filenames
% - labels: nImgs vector: numeric values encoding the class label of each image
% - outFile: string, the filename in which to cache the activations
    if ~exist(outFile,'file')

        % process a single image first to initialize the arrays
        load(files{1},'-mat','c2'); % yes, sadly, they're all called c2
        x = nan(length(files),numel(c2));
        y = zeros(length(files), max(labels)); % non-sparse for python
        x(1,:) = c2;
        y(1,labels(1)) = 1; % use a one-hot encoding of the labels

        % Then, process the rest
        for i = 2:length(files)
            load(files{i},'-mat','c2');
            x(i,:) = c2;
            y(i,labels(i)) = 1;
        end

        % Write the file to disk.
        save(outFile,'x','y','-v7.3','-mat');

    end
end
