function evaluateFeatureSets(p,basetype,tr_data,te_data)
% evaluateFeatureSets(p,basetype,tr_data,te_data)

    % % % build all the stuff we need in advance % % %

    files_tr= tr_data.file;
    files_te= te_data.file;
    labels_tr = tr_data.label;
    labels_te = te_data.label;
    fprintf('training *and* testing image/label lists configured\n');

    files_tr_ge = cell(length(files_tr),1);
    files_tr_ca = cell(length(files_tr),1);
    parfor iFile = 1:length(files_tr)
        [d,f,~] = fileparts(files_tr{iFile});
        files_tr_ge{iFile} = [d '/' f '.' basetype '_gen_mat'];
        files_tr_ca{iFile} = [d '/' f '.' basetype '_cat_mat'];
    end
    fprintf('Built training general *and* conceptual feature cache lists\n');

    files_te_ge = cell(length(files_te),1);
    files_te_ca = cell(length(files_te),1);
    parfor iFile = 1:length(files_te)
        [d,f,~] = fileparts(files_te{iFile});
        files_te_ge{iFile} = [d '/' f '.' basetype '_gen_mat'];
        files_te_ca{iFile} = [d '/' f '.' basetype '_cat_mat'];
    end
    fprintf('Built testing general *and* conceptual feature cache lists\n');

    % % % create the function that actually performs the evaluations % % %

    function evaluationHelper(outStem,m_tr,m_te,labels_tr,labels_te)

        assert(strcmp(p.method,'logreg'),'EvaluateFeatureSets: method should be logistic regression!\n');

        % create cross-validation splits for training with 1,2,4,8,... examples
        % (testing will always use the same images for comparison)
        splitFile = [p.outDir outStem '-splits.mat'];
        if ~exist(splitFile,'file')
            rngState = rng;
            cvsplit = multiclass_cv(labels_tr,p.nTrainingExamples,p.nRuns);
            save(splitFile,'-mat','rngState','cvsplit');
            fprintf('50/50 splits generated for %s\n',outStem);
        else
            load(splitFile,'-mat','cvsplit');
            fprintf('50/50 splits loaded for %s\n',outStem);
        end

        outFile = [p.outDir outStem '-evaluation.csv'];
        if ~exist(outFile,'file')
            options2 = p.options;
            options2.dir = [options2.dir outStem];
            results = evaluatePerformanceAlt(...
              m_tr',labels_tr',m_te',labels_te',cvsplit,options2);
            writetable(results,outFile);
        end
        fprintf('%s evaluated\n',outStem);

    end

    % % % evaluate the general features % % %

    [m_tr_ge,labels_tr_ge] = buildActivationMatrix(files_tr_ge,labels_tr);
    [m_te_ge,labels_te_ge] = buildActivationMatrix(files_te_ge,labels_te);
    fprintf('Built generic activations\n');

    evaluationHelper([basetype '-general'],m_tr_ge,m_te_ge,labels_tr_ge,labels_te_ge);
    clear m_tr_ge m_te_ge labels_tr_ge labels_te_ge;

    % % % evaluate the categorical features % % %

    [m_tr_ca,labels_tr_ca] = buildActivationMatrix(files_tr_ca,labels_tr);
    [m_te_ca,labels_te_ca] = buildActivationMatrix(files_te_ca,labels_te);
    fprintf('Built categorical activations\n');

    evaluationHelper([basetype '-categorical'],m_tr_ca,m_te_ca,labels_tr_ca,labels_te_ca);
    clear m_tr_ca m_te_ca labels_tr_ca labels_te_ca;

%   % % % evaluate the combination of general and categorical features % % %

%   [m_tr_ge,labels_tr_ge] = buildActivationMatrix(files_tr_ge,labels_tr);
%   [m_te_ge,labels_te_ge] = buildActivationMatrix(files_te_ge,labels_te);
%   fprintf('Built generic activations\n');
%   [m_tr_ca,~] = buildActivationMatrix(files_tr_ca,labels_tr);
%   [m_te_ca,~] = buildActivationMatrix(files_te_ca,labels_te);
%   fprintf('Built categorical activations\n');

%   evaluationHelper([basetype '-super'],[m_tr_ge;m_tr_ca],[m_te_ge;m_te_ca],labels_tr_ge,labels_te_ge);

end
