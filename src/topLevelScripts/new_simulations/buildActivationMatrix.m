function outMat = buildActivationMatrix(files,labels,outFile)
% outMat = buildActivationMatrix(files,labels)
%
% given a list of files storing activations and their labels,
% create a matrix of the activations *and* a matrix of one-hot 
% vectors encoding the labels.
%
% labels should be numeric values.
    if ~exist(outFile,'file')
        y = sparse([],[],[],length(files),max(labels),length(files));
        for i = 1:length(files)
            load(files{i},'-mat','c2'); % yes, sadly, they're all called c2
            x(i,:) = c2;
            y(i,labels(i)) = 1;
        end
        save(outFile,'x','y','-v7.3','-mat');
    end
    outMat = matfile(outFile);
end
