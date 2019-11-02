#!/bin/bash

# For each directory in $1, fix a file naming bug.
echo "It begins"
cd $1/
for d in $1/* ; do 
        cd $d
        pwd
        rename 's/\.googlnet_cat_mat/\.googlenet_cat_mat/' *mat
        ls -1 | head -n 1
        cd ..
    done

# Copy all the activations in $1 to /data2/image_sets/image_net/images.
echo "moving everything to data2"
find * -iname "*mat" | xargs -n 1 -P 0 -I % cp % /data2/image_sets/image_net/images/%
