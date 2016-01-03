package cs446.weka.classifiers.trees;


import weka.classifiers.*;
import weka.core.Instances;
import weka.core.Instance;
import java.util.*;
import java.io.IOException;
import java.lang.Boolean;
import java.lang.Exception;
import java.util.Collections;

public class SGD extends Classifier
{	
	double deltaWeight = 0.1;
	double weight0 = 0;
	//Error Definitions
	double Error = 0.0;
	double prevError = 0.0;
	double deltaError = 0.0;
	//Error Threshold
	double threshold = 0.000095;
	
	//Number of Epochs
	double R = 0.0001;
	//Learning Rate
	int epochs = 150;
	boolean trained = false;
	private double[] weightArray;
	
   public void buildClassifier(Instances argF) throws IOException
   {
	   int numFeatures = argF.numAttributes() - 1;
	   if(numFeatures == 100)
	   {
		   //For the Decision Stumps Problem 2.e
		   threshold = 0.000009;
	           R = 0.000036;
		   epochs = 250;
	   }
	   int numInstances = argF.numInstances();
	   weightArray = new double[numFeatures];
	   for(int i = 0; i < numFeatures ; i++)
	   {
		    weightArray[i] = 0.0;
	   }
	   int iter = 0;
	   ArrayList<Integer> indices = new ArrayList<Integer>();
	   for(int nI = 0; nI < argF.numInstances(); nI++)
	        indices.add(nI);
	   
	   deltaError = 10000; //Some large Value to bypass while
	   while((iter < epochs) && ((deltaError) > threshold))
	   {
		   iter++;
		   int i = 0;
		   while(i < indices.size())
		      {
			    Instance tempInstance = argF.instance(i);
			    i++;
			    double[] x_i = new double[numFeatures];
			    double dotProduct = 0;
			    for(int A = 0; A < (numFeatures) ; A++)
			    {
	                           x_i[A] = tempInstance.value(A);
			    }
		            double y_i = tempInstance.classValue();
		        
			    for(int A  = 0; A < (tempInstance.numAttributes() -1) ; A++)
		            {
		        	          dotProduct +=  x_i[A] * weightArray[A]; 
		            }  
		        deltaWeight = y_i - dotProduct;
		        weight0 = weight0 +  R*deltaWeight;
		        for(int A = 0; A < tempInstance.numAttributes() - 1; A++)
		        {
		        	weightArray[A] = weightArray[A] + R*deltaWeight*x_i[A];
		        }
		        prevError = Error;
		        Error += 0.5 * deltaWeight * deltaWeight/numFeatures;  
		      }
		      deltaError = Error - prevError;
		       
	   }
	   
	   trained = true;
	}

   public double classifyInstance(Instance clInstance) throws java.lang.Exception 
   {
		if(!trained)
		{
		    throw new Exception("The classifier has not been trained!");
		}
		double dotProduct = 0.0;
		double result;
		double x_i[] = new double[clInstance.numAttributes()-1];
		for(int A = 0; A < (clInstance.numAttributes() - 1) ; A++)
	    {
                       x_i[A] = clInstance.value(A);
        }
		for(int A  = 0; A < (clInstance.numAttributes() -1) ; A++)
        {
        	          dotProduct +=  x_i[A] * weightArray[A]; 
        }
        result = dotProduct + weight0;
	return (result  >= 0.5) ? 1.0 : 0.0;
   }


}
