% cacheHumanReadableCategoryLists.m
%
% a simple MATLAB script for listing the categories used for
% training/vocabulary concepts and the evaluation concepts in a csv

% key directories
simDir = '/data1/josh/concept_learning/';
imgDir = '/data2/image_sets/image_net/';

% make utility code available
addpath(genpath([simDir 'src']));
addpath(genpath([imgDir 'toolbox']));

% ImageNet's knowledge of itself
srFile = [imgDir 'structure_released.xml'];

% process the training/vocabulary categories
trainInFile = [simDir 'evaluation/v0_2/trainingCategories.csv'];
trainOutFile = [simDir 'evaluation/v0_2/trainingSupplemental.csv']
trainInTable = readtable(trainInFile,'ReadVariableNames',true);
trainWnids = trainInTable.synset;

listImageNetMeanings(trainWnids,srFile,trainOutFile);

% process the evaluation categories
evalInFile1 = [simDir 'evaluation/v0_2/validationCategories.csv'];
evalInFile2 = [simDir 'evaluation/v0_2/subset-of-classes.mat'];
evalOutFile = [simDir 'evaluation/v0_2/validationSupplemental.csv'];
evalInData = load(evalInFile2);
evalInTable = readtable(evalInFile1,'ReadVariableNames',true);
evalWnids = evalInTable.synset[evalInData.classes];

listImageNetMeanings(evalWnids,srFile,evalOutFile);
