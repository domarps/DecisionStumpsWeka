package cs446.weka.classifiers.trees;


import java.io.File;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Scanner;
import java.util.Set;

import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;

public class FeatureGenerator 
{

    static String[] features;
    static String[] numericFeatures = {"firstLength","lastLength"};
    private static FastVector attributes;
    private static Attribute classLabel;	
    private static java.util.Map<String,Attribute> attribute_dict = new java.util.HashMap<String,Attribute>();
    private static FastVector zeroOne;
    private static FastVector labels;
    private static FastVector lengthS;
    static 
    {
	features = new String[] { "firstName", "lastName" };

	List<String> ff = new ArrayList<String>();

	for (String f : features) 
	{
           for(int i = 0; i < 5; i++)
           { 
	    for (char letter = 'a'; letter <= 'z'; letter++) 
	    {
		ff.add(f + i + "=" + letter);
	    }
	   }
	}
/*
	for (String f : features) 
	{
           for(int i = 3; i < 10 ; i++)
           { 
		ff.add(f  + "=" + i);
	   }
	}
*/
	features = ff.toArray(new String[ff.size()]);
        attributes = new FastVector();
	for(String numeric_name : numericFeatures){
		for(int i = 1; i <= 5; i++)
		{
			String att_name = numeric_name + "=" + i;
			Attribute a_len_attribute = new Attribute(att_name, zeroOne);
			attribute_dict.put(att_name, a_len_attribute);
			attributes.addElement(a_len_attribute);
			ff.add(att_name);
		}
	}
	
	zeroOne = new FastVector(2);
	zeroOne.addElement("1");
	zeroOne.addElement("0");

	labels = new FastVector(2);
	labels.addElement("+");
	labels.addElement("-");

	lengthS = new FastVector(7);
	lengthS.addElement("3");
	lengthS.addElement("4");
	lengthS.addElement("5");
	lengthS.addElement("6");
	lengthS.addElement("7");
	lengthS.addElement("8");
	lengthS.addElement("9");
	lengthS.addElement("10");
    }

    public static Instances readData(String fileName) throws Exception 
    {

	Instances instances = initializeAttributes();
	Scanner scanner = new Scanner(new File(fileName));

	while (scanner.hasNextLine()) {
	    String line = scanner.nextLine();

	    Instance instance = makeInstance(instances, line);

	    instances.add(instance);
	}

	scanner.close();

	return instances;
    }

    private static Instances initializeAttributes() 
    {

	String nameOfDataset = "Badges";

	Instances instances;

	FastVector attributes = new FastVector(9);
	for (String featureName : features) 
	{
	    attributes.addElement(new Attribute(featureName, zeroOne));
        }

	for (String featureName : numericFeatures) 
	{
	    attributes.addElement(new Attribute(featureName, zeroOne));
        }
	
	//labels is a FastVector of '+' and '-'
//	classLabel = new Attribute("Class", labels);
//	attributes.addElement(classLabel);
         
	instances = new Instances(nameOfDataset, attributes, 0);

	instances.setClass(classLabel);

	return instances;

    }

    private static Instance makeInstance(Instances instances, String inputLine) 
    { 
	String MagicWord = "danrothsmachinelearning";
	inputLine = inputLine.trim();
        //We need to store the lastName as well...
	String[] parts = inputLine.split("\\s+");
	String label = parts[0];
	String firstName = parts[1].toLowerCase();
        String lastName  = parts[2].toLowerCase();

	Instance instance = new Instance(features.length + 1);
	instance.setDataset(instances);

	Set<String> feats = new HashSet<String>();
	/*
	feats.add("firstName0=" + firstName.charAt(0));
	feats.add("firstNameN=" + firstName.charAt(firstName.length() - 1));
        */
	for(int f = 0; f < 8 ; f++)
	{
           if(firstName.length() > f)// && MagicWord.indexOf(firstName.charAt(f)) != -1)
            	  feats.add("firstName"+f+"="+firstName.charAt(f));
	}
        for(int l = 0; l < 8; l++)
        {
	   if(lastName.length() > l) // && MagicWord.indexOf(lastName.charAt(l)) != -1)
		  feats.add("lastName"+l+"="+lastName.charAt(l));
	}	
	/////////////////////////////////////////////////////////////////
	for (int featureId = 0; featureId < features.length; featureId++) {
	    Attribute att = instances.attribute(features[featureId]);

	    String name = att.name();
	    String featureLabel;
	    if (feats.contains(name)) {
		featureLabel = "1";
	    } else
		featureLabel = "0";
	    instance.setValue(att, featureLabel);
	}

	instance.setValue(attribute_dict.get("first_len=" + Math.min(firstName.length(),5)), "1");
	instance.setValue(attribute_dict.get("last_len=" + Math.min(lastName.length(), 5)), "1");
	instance.setClassValue(label);

	return instance;
    }

    public static void main(String[] args) throws Exception 
    {

	if (args.length != 2) {
	    System.err
		    .println("Usage: FeatureGenerator input-badges-file features-file");
	    System.exit(-1);
	}
	Instances data = readData(args[0]);
	ArffSaver saver = new ArffSaver();
	saver.setInstances(data);
	saver.setFile(new File(args[1]));
	saver.writeBatch();
   }
}
