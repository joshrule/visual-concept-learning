#!/bin/bash

# In several directories, fix a file naming bug.

cd /data2/image_sets/image_net/ILSVRC2015/Data/CLS-LOC/val/
echo `pwd`
rename 's/.googlenet_cat_mat.mat$/.googlenet_cat_mat/' *.googlenet_cat_mat.mat
rename 's/.googlenet_gen_mat.mat$/.googlenet_gen_mat/' *.googlenet_gen_mat.mat

for dir in /data2/image_sets/image_net/ILSVRC2015/Data/CLS-LOC/train/*/
do
    cd $dir
    echo `pwd`
    rename 's/.googlenet_cat_mat.mat$/.googlenet_cat_mat/' *.googlenet_cat_mat.mat
    rename 's/.googlenet_gen_mat.mat$/.googlenet_gen_mat/' *.googlenet_gen_mat.mat
done
