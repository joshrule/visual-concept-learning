function trainModels(caffe_dir)
% trainModels(caffe_dir)
%
% Train the Caffe models needed for the simulations.
%
% caffe_dir: string, the directory containing the Caffe models & binaries.

    % go to the caffe dir & launch some training!
    cd(caffe_dir);

    % % ignoring HMAX results for now!
    % % train HMAX model
    % if ~exist([caffe_dir 'models/hmax_softmax/trained.flag'],'file')
    %     system(['touch ' caffe_dir 'models/hmax_softmax/trained.flag']);
    %     system(['./build/tools/caffe train ' ...
    %             '--solver=models/hmax_softmax/solver.prototxt ' ...
    %             '--gpu=all ' ...
    %             '> models/hmax_softmax/training.log 2>&1 &']);
    % end

    % train GoogLeNet
    if ~exist([caffe_dir 'models/maxlab_googlenet/training.log'],'file')
        system(['./build/tools/caffe train ' ...
                '--solver=models/maxlab_googlenet/solver.prototxt ' ...
                '--gpu=all ' ...
                '> models/maxlab_googlenet/training.log 2>&1 &']);
    end

    % test GoogLeNet
    if ~exist([caffe_dir 'models/maxlab_googlenet/testing.log'],'file')
        system(['./build/tools/caffe test '...
                '-model models/maxlab_googlenet/train_val.prototxt '...
                '-weights models/maxlab_googlenet/maxlab_googlenet_iter_10000000.caffemodel '
                '-gpu 0 -iterations 6250 > models/maxlab_googlenet/testing.log 2>&1']);
    end
end
