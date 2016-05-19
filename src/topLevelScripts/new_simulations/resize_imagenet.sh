#!/bin/bash

# TODO:
# - 256x256 is taken from the main simulation parameters
# - hardcoded directories

echo -e "*********\nIt Begins\n*********"

IMGHOME="/data2/image_sets/image_net"
CAFFEBIN="/home/josh/caffe/build/tools"
SIMHOME="/data1/josh/ruleRiesenhuber2013"
MATLAB="/home/josh/bin/MATLAB/R2015b/bin/matlab"

#echo "resizing all the images"
#DIR1="$IMGHOME/images"
#echo -e "\tworking in $DIR1"
#find $DIR1 -iname "*.JPEG" | xargs -I % -n 1 -P 32 convert % -resize 256x256^ %
#
#DIR2="$IMGHOME/ILSVRC2015/Data"
#echo -e "\tworking in $DIR2"
#find $DIR2 -iname "*.JPEG" | xargs -I % -n 1 -P 32 convert % -resize 256x256^ %

LIST2="$SIMHOME/caffe/feature_validation_images.txt"
DB2="$IMGHOME/feat_val_lmdb"
echo -e "\tworking on $DB2"
$CAFFEBIN/convert_imageset / $LIST2 $DB2
$MATLAB -nodesktop -nodisplay -nosplash -r "compute_means('$LIST2'); exit;"

LIST4="$SIMHOME/caffe/evaluation_validation_images.txt"
DB4="$IMGHOME/eval_val_lmdb"
echo -e "\tworking on $DB4"
$CAFFEBIN/convert_imageset / $LIST4 $DB4
$MATLAB -nodesktop -nodisplay -nosplash -r "compute_means('$LIST4'); exit;"

LIST3="$SIMHOME/caffe/evaluation_training_images.txt"
DB3="$IMGHOME/eval_train_lmdb"
echo -e "\tworking on $DB3"
$CAFFEBIN/convert_imageset / $LIST3 $DB3
$MATLAB -nodesktop -nodisplay -nosplash -r "compute_means('$LIST3'); exit;"

LIST1="$SIMHOME/caffe/feature_training_images.txt"
DB1="$IMGHOME/feat_train_lmdb"
echo -e "\tworking on $DB1"
$CAFFEBIN/convert_imageset / $LIST1 $DB1
$MATLAB -nodesktop -nodisplay -nosplash -r "compute_means('$LIST1'); exit;"
