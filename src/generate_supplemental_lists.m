simDir = '/data1/josh/concept_learning/';
imgDir = '/data2/image_sets/image_net/';

addpath(genpath([simDir 'src']));
addpath(genpath([imgDir 'toolbox']));

srFile = [imgDir 'structure_released.xml'];

trainInFile = [simDir 'evaluation/v0_2/trainingCategories.csv'];
trainOutFile = [simDir 'evaluation/v0_2/trainingSupplemental.csv']
trainInTable = readtable(trainInFile,'ReadVariableNames',true);
trainWnids = trainInTable.synset;

listImageNetMeanings(trainWnids,srFile,trainOutFile);

evalInFile1 = [simDir 'evaluation/v0_2/validationCategories.csv'];
evalInFile2 = [simDir 'evaluation/v0_2/subset-of-classes.mat'];
evalOutFile = [simDir 'evaluation/v0_2/validationSupplemental.csv'];
evalInData = load(evalInFile2);
evalInTable = readtable(evalInFile1,'ReadVariableNames',true);
evalWnids = evalInTable.synset[evalInData.classes];

listImageNetMeanings(evalWnids,srFile,evalOutFile);
