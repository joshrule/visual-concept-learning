function p = Linear5050Params()
    home = '/data1/josh/concept_learning/';
    suffix = 'v0_2';
    imgNetHome = '/data2/image_sets/image_net/';

    p = struct( ...
        'home', home, ...
        'caffe_dir', '/home/josh/caffe/', ...
        'srcPaths', {{[home 'src/'],[imgNetHome 'toolbox/']}}, ...
        'seed', 0, ...
        'imgNetDir', [imgNetHome 'images/'], ...
        'imgNetValDir', [imgNetHome 'ILSVRC2015/'], ...
        'nImgs', 732, ...
        'nCategories', 2000, ...
        'validationCategoryFile', [imgNetHome 'lists/ilsvrc_obj_loc_train_categories.txt'], ...
        'allCategoryFile', [imgNetHome 'lists/synset_counts.txt'], ...
        'srFile', [imgNetHome 'images/structure_released.xml'], ...
        'imgNetUser', 'joshrule', ...
        'imgNetKey', '8f198f78448a0674277158b595778f55b13ed432', ...
        'suffix', suffix, ...
        'outDir', ensureDir([home 'evaluation/' suffix '/']), ...
        'nTrValidationImgs',25, ...
        'featDir', [home 'image_sets/image_net/'], ...
        'maxSize', 256, ...
        'hmaxHome', [home 'src/hmax-ocv/'], ...
        'patchFiles', {{[home 'patches/kmeans.xml']}}, ...
        'modelDir', [home 'patches/c3vLinear/'], ...
        'method', 'logreg', ...
        'options', struct( ...
           'dir', [home 'evaluation/' suffix '/evaluation_data/'], ...
           'N', 16000, ...
           'nGPUs', 4), ...
        'nTrainingExamples', [1 2 4 8 16 32 64 128 256 512], ...
        'nBinaryTrainingExamples', [1 2 4 8 16 32 64], ...
        'nCategoricityTrainingExamples', 64, ...
        'nBinaryCategories', 100, ...
        'nRuns', 20, ...
        'testingThreshes', [50 100 200 500 1000 2000], ...
        'nValidationRuns', 10);
end

function success = ensureDir(dirName)
% Author: Santosh Divvala
% Revised: saurabh.me@gmail.com (Saurabh Singh).
% Revised: rsj28@georgetown.edu (Josh Rule).
%
% Conditionally create a directory and return its path
%
% dirName: string, the absolute path of the directory
%
% success: string, the absolute path of the directory, empty on failure
    if exist(dirName, 'dir') || mkdir(dirName)
        success = dirName;
    else
        success = '';
    end
end
