function [x,y] = buildActivationMatrix(files,labels)
% [x,y] = buildActivationMatrix(files,labels)
%
% given a list of files storing activations and their labels,
% create a matrix of the activations *and* a matrix of one-hot 
% vectors encoding the labels.
%
% labels should be numeric values.
    y = sparse([],[],[],max(labels),length(files),length(files));
    for i = 1:length(files)
        load(files{i},'-mat','c2'); % yes, sadly, they're all called c2
        x(:,i) = c2;
        y(labels(i),i) = 1;
    end
end
