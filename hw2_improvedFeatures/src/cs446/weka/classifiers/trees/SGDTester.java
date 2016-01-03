package cs446.weka.classifiers.trees;

import java.io.File;
import java.io.FileReader;

import weka.classifiers.Evaluation;
import weka.core.Instances;

public class SGDTester
{
   public static void main(String[] args)throws Exception
	  {
         if(args.length != 1)
         {
            	 System.out.println("Correct Usage : SGDTester arff-file");
            	 System.exit(-1);
	     }
	     Instances data = new Instances(new FileReader(new File(args[0])));
         data.setClassIndex(data.numAttributes() - 1);
             	     
         int numFolds = 5;
         double sum_accuracy = 0;
         double total_accuracy = 0;
         
	      for(int i = 0; i < numFolds;i++)
         {
              SGD classifierTrainer = new SGD();
	      Instances train = data.trainCV(numFolds,i);

	      System.out.println("Results for the " + (i+1) +"th Fold ");
	      classifierTrainer.buildClassifier(train);
	      //System.out.println("Number of instances in Complete Data " +data.numInstances());
	     //System.out.println("Number of instances in Training Set  " +train.numInstances());
	      
              Instances test  = data.testCV(numFolds,i);
          //System.out.println("Number of instances in Test Set  "+test.numInstances());
          Evaluation evaluation = new Evaluation(test);
	      evaluation.evaluateModel(classifierTrainer,test);
	      System.out.println(evaluation.toSummaryString());
	      sum_accuracy   += evaluation.correct();
	      total_accuracy += evaluation.correct() + evaluation.incorrect();
	      
	  }
      //System.out.println(evaluation.toSummaryString());
	         
	  System.out.println("Average percentage across five fold cross validation:");
	  System.out.println(sum_accuracy/total_accuracy * 100 + "%");
	  
	  
	  
	  //SGD classifierFullTrain = new SGD();
	  //classifierFullTrain.buildClassifier(data);
	  
      	  
}
}
