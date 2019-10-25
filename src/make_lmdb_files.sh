#!/bin/bash

# TODO: lots of hardcoded values 

echo -e "making the LMDB files"

IMGHOME="/data2/image_sets/image_net"
LMDBHOME="/data3/image_net_lmdbs"
CAFFEBIN="/home/josh/caffe/build/tools"
SIMHOME="/data1/josh/concept_learning"
MATLAB="/home/josh/bin/MATLAB/R2015b/bin/matlab"

if [ ! -f $SIMHOME/evaluation/v0_2/resized.flag ]; then
    echo "resizing all the images"
    DIR1="$IMGHOME/images"
    echo -e "working in $DIR1"
    find $DIR1 -iname "*.JPEG" | xargs -I % -n 1 -P 32 convert % -resize 256x256^ %

    DIR2="$IMGHOME/ILSVRC2015/Data"
    echo -e "working in $DIR2"
    find $DIR2 -iname "*.JPEG" | xargs -I % -n 1 -P 32 convert % -resize 256x256^ %

    touch $SIMHOME/evaluation/v0_2/resized.flag
fi

DB2="$LMDBHOME/feat_val_lmdb"
LIST2="$SIMHOME/caffe/feature_validation_images.txt"
echo -e "working on $DB2"
if [ ! -d $DB2 ]; then
    $CAFFEBIN/convert_imageset -shuffle / $LIST2 $DB2
fi
if [ ! -f ${LIST2%.txt}_means.csv ]; then
    $MATLAB -nodesktop -nodisplay -nosplash -r "computeMeans('$LIST2'); exit;"
fi

LIST4="$SIMHOME/caffe/evaluation_validation_images.txt"
DB4="$LMDBHOME/eval_val_lmdb"
echo -e "working on $DB4"
if [ ! -d $DB4 ]; then
    $CAFFEBIN/convert_imageset -shuffle / $LIST4 $DB4
fi
if [ ! -f ${LIST4%.txt}_means.csv ]; then
    $MATLAB -nodesktop -nodisplay -nosplash -r "computeMeans('$LIST4'); exit;"
fi

LIST3="$SIMHOME/caffe/evaluation_training_images.txt"
DB3="$LMDBHOME/eval_train_lmdb"
echo -e "working on $DB3"
if [ ! -d $DB3 ]; then
    $CAFFEBIN/convert_imageset -shuffle / $LIST3 $DB3
fi
if [ ! -f ${LIST3%.txt}_means.csv ]; then
    $MATLAB -nodesktop -nodisplay -nosplash -r "computeMeans('$LIST3'); exit;"
fi

LIST1="$SIMHOME/caffe/feature_training_images.txt"
DB1="$IMGHOME/feat_train_lmdb"
echo -e "working on $DB1"
if [ ! -d $DB1 ]; then
    $CAFFEBIN/convert_imageset -shuffle / $LIST1 $DB1
fi
if [ ! -f ${LIST1%.txt}_means.csv ]; then
    $MATLAB -nodesktop -nodisplay -nosplash -r "computeMeans('$LIST1'); exit;"
fi
