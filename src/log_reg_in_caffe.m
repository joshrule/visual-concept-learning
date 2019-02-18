function [classificationValues,model] = log_reg_in_caffe(trainX,trainY,testX,testY,options)
% This function takes arguments for doing multinomial logistic regression and
% regresses in Caffe. It returns the classification values and the model.

    final_dir = options.dir;
    if ~exist([final_dir 'results.mat'],'file')
        % convert the training labels
        [trainY,~] = ind2sub(size(trainY'),find(trainY'));
        [testY,~] = ind2sub(size(testY'),find(testY'));

        % shuffle the data
        reorder_train = randperm(length(trainY));
        trainX = trainX(reorder_train,:);
        trainY = trainY(reorder_train)-1; % subtract one for Python's 0-indexing
        reorder_test = randperm(length(testY));
        testX = testX(reorder_test,:);
        testY = testY(reorder_test)-1; % subtract one for Python's 0-indexing

        % save the training/testing data
        ensureDir(final_dir);
        save([final_dir 'data.mat'],'trainX','trainY','testX','testY','reorder_train','reorder_test');
        clear trainX trainY testX testY;

        % run the classifier
        system(['LD_LIBRARY_PATH=/usr/lib/:/usr/local/lib/:/usr/local/cuda/lib:/usr/local/cuda/lib64 ' ...
                'python mnlr2.py ' num2str(options.gpu) ' ' final_dir ' &> ' final_dir 'log.log']);
    else
        load([final_dir 'data.mat'],'reorder_test');
    end

    % report the results
    load([final_dir 'results.mat']);
    [~,deorder_test] = sort(reorder_test,'ascend'); % deshuffle the data for return
    classificationValues = classificationValues(deorder_test,:);
    model = final_dir;
end
