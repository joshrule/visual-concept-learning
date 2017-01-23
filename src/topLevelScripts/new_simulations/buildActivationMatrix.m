function [x,y] = buildActivationMatrix(files,categories)
    cat_index = sort(unique(categories));
    y = zeros(length(cat_index),length(files));
    for i = 1:length(files)
        load(files{i},'-mat','c2'); % yes, sadly, they're all called c2
        idx = find(ismember(cat_index,categories{i}));
        x(:,i) = c2;
        y(idx,i) = 1;
    end
end
