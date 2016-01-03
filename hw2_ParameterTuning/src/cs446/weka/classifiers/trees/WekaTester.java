package cs446.weka.classifiers.trees;

import java.io.File;
import java.io.FileReader;

import weka.classifiers.Evaluation;
import weka.core.Instances;

public class WekaTester {

    public static void main(String[] args) throws Exception {

	if (args.length != 1) {
	    System.err.println("Usage: WekaTester arff-file");
	    System.exit(-1);
	}

	// Load the data
	Instances data = new Instances(new FileReader(new File(args[0])));

	// The last attribute is the class label
	data.setClassIndex(data.numAttributes() - 1);
    double totalCorrect = 0;
    double totalPossible = 0;
    int numFolds = 5;
	// Train on 80% of the data and test on 20%
	for(int i = 0; i < numFolds ; i++)
	{
		System.out.println("Fold" + i);
		Instances train = data.trainCV(numFolds,i);
		Instances test = data.testCV(numFolds, i);
	
		// Create a new ID3 classifier. This is the modified one where you can
		// set the depth of the tree.
		Id3 classifierID3 = new Id3();

		// An example depth. If this value is -1, then the tree is grown to full
		// depth.
		classifierID3.setMaxDepth(-1);

		// Train
		classifierID3.buildClassifier(train);

		// Print the classfier
		System.out.println(classifierID3);
		System.out.println();

		// Evaluate on the test set
		Evaluation evaluation = new Evaluation(test);
		evaluation.evaluateModel(classifierID3, test);
		System.out.println(evaluation.toSummaryString());
		totalCorrect += evaluation.correct();
		totalPossible += evaluation.correct() + evaluation.incorrect();
	}
	System.out.println("Average percentage across five-fold cross validation");
	System.out.println(totalCorrect/totalPossible * 100 + "%");
   }
}
