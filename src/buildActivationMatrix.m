function outFile = buildActivationMatrix(files,labels,outFile)
% outMat = buildActivationMatrix(files,labels)
%
% given a list of files storing activations and their labels,
% create a matrix of the activations *and* a matrix of one-hot 
% vectors encoding the labels.
%
% labels should be numeric values.
    if ~exist(outFile,'file')
        fprintf('0...');

        load(files{1},'-mat','c2'); % yes, sadly, they're all called c2
        x = nan(length(files),numel(c2));
        x(1,:) = c2;

        % since I load these in python, let's not make them sparse
        y = zeros(length(files), max(labels));
        y(1,labels(1)) = 1;

        for i = 2:length(files)
            if mod(i,1000) == 0
                fprintf('%d...',i);
            end
            load(files{i},'-mat','c2');
            x(i,:) = c2;
            y(i,labels(i)) = 1;
        end
        fprintf('%d!\n', length(files));
        save(outFile,'x','y','-v7.3','-mat');
    end
end
