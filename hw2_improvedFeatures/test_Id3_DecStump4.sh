#!/bin/bash

mkdir bin

make
java -cp lib/weka.jar:bin cs446.weka.classifiers.trees.FeatureGenerator data/badges.modified.data.all data/myBadges.arff
java -cp lib/weka.jar:bin cs446.weka.classifiers.trees.Id3DecStump4Tester data/myBadges.arff
