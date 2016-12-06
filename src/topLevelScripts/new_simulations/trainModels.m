function trainModels(caffe_dir)

    % go to the caffe dir & launch some training!
    cd(caffe_dir);

    % start one
    if ~exist([caffe_dir 'models/hmax_softmax/trained.flag'],'file')
        system(['touch ' caffe_dir 'models/hmax_softmax/trained.flag']);
        system(['./build/tools/caffe train ' ...
                '--solver=models/hmax_softmax/solver.prototxt ' ...
                '--gpu=all ' ...
                '> models/hmax_softmax/training.log 2>&1 &']);
    end

% % % trained prior to writing this code
%   % then, the other
%   if ~exist([caffe_dir 'models/maxlab_googlenet/trained.flag'],'file')
%       system(['touch ' caffe_dir 'models/maxlab_googlenet/trained.flag']);
%       system(['./build/tools/caffe train ' ...
%               '--solver=models/maxlab_googlenet/solver.prototxt ' ...
%               '--gpu=all ' ...
%               '> models/maxlab_googlenet/training.log 2>&1 &']);
%   end

    if ~exist([caffe_dir 'models/maxlab_googlenet/testing.log'],'file')
        system(['./build/tools/caffe test '...
                '-model models/maxlab_googlenet/train_val.prototxt '...
                '-weights models/maxlab_googlenet/maxlab_googlenet_iter_10000000.caffemodel '
                '-gpu 0 -iterations 6250 > models/maxlab_googlenet/testing.log 2>&1']);
    end
end
