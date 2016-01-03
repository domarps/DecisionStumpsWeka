#!/bin/bash

mkdir bin

make

# Generate the example features (first and last characters of the
# first names) from the entire dataset. This shows an example of how
# the featurre files may be built. Note that don't necessarily have to
# use Java for this step.

java -cp lib/weka.jar:bin cs446.weka.classifiers.trees.FeatureGenerator data/badges.modified.data.all data/myBadges.arff

# Using the features generated above, train a decision tree classifier
# to predict the data. This is just an example code and in the
# homework, you should perform five fold cross-validation. 
#java -cp lib/weka.jar:bin cs446.weka.classifiers.trees.WekaTester data/myBadges.arff
#java -cp lib/weka.jar:bin cs446.weka.classifiers.trees.SGDTester data/myBadges.arff
#java -cp lib/weka.jar:bin cs446.weka.classifiers.trees.Id3Tester data/myBadges.arff
#java -cp lib/weka.jar:bin cs446.weka.classifiers.trees.SGDTester data/myBadges.arff
#java -cp lib/weka.jar:bin cs446.weka.classifiers.trees.Id3Tester data/myBadges.arff

