#!/bin/bash

# for each directory in arg1
echo "It begins"
cd $1/
for d in $1/* ; do 
        cd $d
        pwd
        rename 's/\.googlnet_cat_mat/\.googlenet_cat_mat/' *mat
        ls -1 | head -n 1
        cd ..
    done

echo "moving everything to data2"
find * -iname "*mat" | xargs -n 1 -P 0 -I % cp % /data2/image_sets/image_net/images/%
