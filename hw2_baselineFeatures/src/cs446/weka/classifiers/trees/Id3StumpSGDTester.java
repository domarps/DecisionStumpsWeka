package cs446.weka.classifiers.trees;
import java.io.File;
import java.io.FileReader;
import java.io.PrintWriter;
import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
public class Id3StumpSGDTester 
{

	
	public static void main(String[] args) throws Exception 
    {
    	if (args.length != 1) 
    	{
	     System.err.println("Usage: WekaTester arff-file");
	     System.exit(-1);
    	}
    	//Initialize Attributes
    	FastVector zeroOne = new FastVector(2);
        zeroOne.addElement("1");
        zeroOne.addElement("0");
           
        FastVector labels = new FastVector(2);
        labels.addElement("0");
        labels.addElement("1");

        FastVector attributes = new FastVector(101);
        for(int c = 0; c < 100; c++) 
        {
    	    Attribute newAttr = new Attribute("DecisionStump"+c, zeroOne);
    	    attributes.addElement(newAttr);
    	}
        Attribute classLabel = new Attribute("Class", labels);
        attributes.addElement(classLabel);
        
        //End Initialize Attributes

    	// Load the data
    	Instances data = new Instances(new FileReader(new File(args[0])));
    	// The last attribute is the class label
    	data.setClassIndex(data.numAttributes() - 1);
    	
    	Id3[] decisionStumps4 = new Id3[100];
        double sum_accuracy = 0;
	    double total_accuracy = 0;
        
	    Evaluation evaluation = new Evaluation(data);
	    //long seed = 1234567891;
	    int numFolds = 5;
	    for(int i = 0; i < numFolds ; i++)
        {
	           Instances train = data.trainCV(numFolds,i);
	           Instances test = data.testCV(numFolds,i);
	           ArrayList<Integer> indices = new ArrayList<Integer>();
	           //Loading indices for the shuffle operation
		       for(int nI = 0; nI < train.numInstances(); nI++)
			        indices.add(nI);
		       Instances sampledTrain = new Instances(train,(train.numInstances()/2));
		       for(int k= 0; k < 100; k++)
	           {
	              sampledTrain.delete();
	              Collections.shuffle(indices);
	              for(int m = 0; m < (indices.size())/2 ; m++)
	              {
	                  sampledTrain.add(train.instance(indices.get(m)));
	              }
	              decisionStumps4[k] = new Id3();
	              decisionStumps4[k].setMaxDepth(4);
	              decisionStumps4[k].buildClassifier(sampledTrain);
		       }
	           //decisionStumps4 is an array of 100 Decision Trees of maxDepth4
		       
		       System.out.println("$$$$$$$Results for the " + (i+1) +"th Fold    $$$$");           
	           Instances InputToSGDTrain = new Instances("DecisionStump4Train",attributes,train.numInstances());
	           InputToSGDTrain.setClass(classLabel);

	           Instances InputToSGDTest = new Instances("DecisionStump4Test",attributes,test.numInstances());
	           InputToSGDTest.setClass(classLabel);
	         
	           for(int t = 0; t < train.numInstances(); t++)
	           {
	        	   Instance curr = train.instance(t);
	        	   Instance OutputInstance = new Instance(101); 
	        	   double featureFromStump;
	        	   for (int j = 0; j < 100; j++)
	        	   {
	        		   featureFromStump =  decisionStumps4[j].classifyInstance(curr);
	        		   OutputInstance.setValue((Attribute)attributes.elementAt(j),featureFromStump);
	               }
	        	   OutputInstance.setValue((Attribute)attributes.elementAt(100),curr.classValue());
	        	   InputToSGDTrain.add(OutputInstance);
	           }
	         
	           for(int t = 0; t < test.numInstances(); t++)
	           {
	        	   Instance curr = test.instance(t);
	        	   Instance OutputInstance = new Instance(101); 
	        	   double featureFromStump;
	        	   for (int j = 0; j < 100; j++)
	        	   {
	        		   featureFromStump =  decisionStumps4[j].classifyInstance(curr);
	        		   OutputInstance.setValue((Attribute)attributes.elementAt(j),featureFromStump);
	               }
	        	   OutputInstance.setValue((Attribute)attributes.elementAt(100),curr.classValue());
	        	   InputToSGDTest.add(OutputInstance);
	           }
	           SGD classifierTrainer = new SGD();
	           classifierTrainer.buildClassifier(InputToSGDTrain);	          
	           evaluation = new Evaluation(InputToSGDTest);
	           evaluation.evaluateModel(classifierTrainer,InputToSGDTest);
	           System.out.println(evaluation.toSummaryString());
	           sum_accuracy   += evaluation.correct();
	 	       total_accuracy += evaluation.correct() + evaluation.incorrect();
        }
	System.out.println("Average percentage across five fold cross validation:");
        System.out.println(sum_accuracy/total_accuracy * 100 + "%");
    }
}
