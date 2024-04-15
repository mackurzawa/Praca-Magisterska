/*
 * Created on 2007-07-05
 * 
 */

package pl.poznan.put.cs.idss.ml.ender;

import java.util.Arrays;
import java.util.Random;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

import pl.poznan.put.cs.idss.ml.*;
import pl.poznan.put.cs.idss.ml.ender.*;
import pl.poznan.put.cs.idss.ml.lossFunctions.*;

import weka.classifiers.*;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;


/**
 * @author wk
 * @author kd
 */

public class Regender extends AbstractClassifier {

	void writeLog(String s){
		try{
			FileWriter writer = new FileWriter("C:\\Users\\Maciej\\Desktop\\Praca Magisterska\\pila.txt", true);
			writer.write(s);
			writer.write("\n");
			writer.close();
		} catch(IOException e){}
	}
	void writeLogArray(double[] s){
		try{
			FileWriter writer = new FileWriter("C:\\Users\\Maciej\\Desktop\\Praca Magisterska\\pila.txt", true);
	
			for (double value : s) {
				writer.write(String.valueOf(value));
				writer.write(" ");
			}
			writer.write("\n...\n");
			writer.close();
		} catch(IOException e){
	
		}
	}
	
	
	//covered instances 
	short[] coveredInstances = null;
	
	// decision rules
	protected Rule[] rules;
	
	// filter for binarization of nominal attributes
	protected NominalToBinary ntb;
		
	ReplaceMissingValues rmv;
	
	// instances
	protected Instances instances;
	
	// number of instances
	protected int numberOfInstances = 0;

	// number of condition attributes
	protected int numberOfConditionAttributes = 0;
		
	// number of decision classes
	protected int numberOfDecisionClasses = 0;
	
	// number of rules to be generated
	protected int M = 0;
	
	// decision attribute index
	protected int decisionAttribute = 0;
	
	//	auxiliary decision attribute index
	protected int auxiliaryDecisionAttribute = 0;
	
	// auxiliary attributes that gives index of an example
	protected int instanceIndexAttribute = 0;
	
	// value of default rule
	protected double defaultRule = 0.0;
	
	// current function values
	protected double[] valueOfF;
	
	protected RuleBuilder ruleBuilder = null;
	
	// whether resampling is performed
	protected boolean resample = false;
	
	// percentage of resampled data
	protected double percentage = 1.0;
	
	// whether resampling is performed with replacement or without 
	protected boolean withReplacement = false;

	// random number generator
	protected Random mainRandomGenerator = null;
	
	private boolean replaceMissingValues = false;
	
	public Instances getInstances() {
		return this.instances;
	}

	public Rule[] getRules() {
		return this.rules;
	}
	
	public double getDefaultRule() {
		return this.defaultRule;
	}
	
	public void setM(int M) {
		this.M = M;
	}
	
	public int getM() {
		return this.M;
	}
	
	public double getValueOfF(int position) {
		return this.valueOfF[position];
	}
	
	public int getNumberOfConditionAttributes() {
		return this.numberOfConditionAttributes;
	}

	public int getDecisionAttribute() {
		return this.decisionAttribute;
	}
	
	public int getAuxiliaryDecisionAttribute() {
		return this.auxiliaryDecisionAttribute;
	}

	public int getInstanceIndexAttribute() {
		return this.instanceIndexAttribute;
	}
	
	public int getNumberOfDecisionClasses() {
		return this.numberOfDecisionClasses;
	}
	
	public boolean ifResample() {
		return this.resample;
	}

	public void setReplaceMissingValues(boolean replaceMissingValues) {
		this.replaceMissingValues = replaceMissingValues;
	}

	public boolean isReplaceMissingValues() {
		return replaceMissingValues;
	}
	
	public Regender(int M, RuleBuilder ruleBuilder, boolean resample, double percentage, boolean withReplacement) {
		this.M = M;
		this.ruleBuilder = ruleBuilder;
		this.resample = resample;
		this.percentage = percentage;
		this.withReplacement = withReplacement;
	}
	
	public String globalInfo() {
		return "Class for building and using an Ensemble of Decision Rules (ENDER).";	    
	}
	
	public short[] resample (int numberOfInstances, double percentage, boolean withReplacement) {
		
		short [] subSample = new short [numberOfInstances];
		
		int subsampleSize = (int) (numberOfInstances * percentage);
		
		if (subsampleSize > 0) {
			if (withReplacement == false) {
				
				Random random = new Random(mainRandomGenerator.nextInt());
				
				int[] indices = new int[numberOfInstances];
				for (int i = 0; i < numberOfInstances; i++)
					indices[i] = i;
				
			    for (int i = numberOfInstances - 1; i > 0; i--) {
			        int temp = indices[i];
			        int index = random.nextInt(i+1);
			        indices[i] = indices[index];
			        indices[index] = temp;
			    }

				for (int i = 0; i < subsampleSize; i++)
					subSample[indices[i]] = 1;				
				
			} else {
				
				Random random = new Random(mainRandomGenerator.nextInt());
				for (int i = 0; i < subsampleSize; i++)
					subSample[random.nextInt(numberOfInstances)]++;
			}
		}
		
		return subSample;
	}

	public void buildClassifier(Instances instances) throws Exception {
		long currentTimeMillis;
		long currentTimeMillisDiff;
		currentTimeMillis = System.currentTimeMillis();  
		
		initialize(instances);
		
		this.rules = new Rule[M];
		
		//create default rule
		Arrays.fill(coveredInstances, (short) 1);
		this.defaultRule = this.ruleBuilder.createDefaultRule(this.valueOfF, this.coveredInstances);
		this.updateFunction(this.defaultRule);

		// writeLog(String.valueOf(this.defaultRule));
		
		int i = 0;
		while (i < M) {
			// writeLog("NEXT RULE ITERATION");
			//resampling
			if(this.ifResample() == true)
				this.coveredInstances = this.resample(this.instances.numInstances(),this.percentage, this.withReplacement);
			else
				Arrays.fill(this.coveredInstances, (short) 1);
			//create rule
			this.rules[i] = this.ruleBuilder.createRule(this.valueOfF, this.coveredInstances);
			if (this.rules[i] != null) {
				//update function F
				this.updateFunction(this.rules[i].getDecision());
				// writeLog("ValueOfF");
				// writeLogArray(valueOfF);
				i++;
			}
			else {
				M = i;
				break;
			}
			// writeLog("\n");
		}
	}

	private void initialize(Instances instances) throws Exception{
		this.instances = new Instances(instances);
		
		ntb = new NominalToBinary();
		try {
			String str = ntb.getAttributeIndices();
			ntb.setInputFormat(this.instances);
			this.instances = Filter.useFilter(this.instances, ntb);
			if (replaceMissingValues) {
				rmv = new ReplaceMissingValues();
				rmv.setInputFormat(this.instances);
				this.instances = Filter.useFilter(this.instances, rmv);
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		this.numberOfConditionAttributes = this.instances.numAttributes() - 1;
		this.numberOfInstances = this.instances.numInstances();
		
		this.valueOfF = new double[this.numberOfInstances];
		Arrays.fill(this.valueOfF, 0.0);
				
		this.decisionAttribute = this.instances.classIndex();
		this.numberOfDecisionClasses = this.instances.numClasses();

		if(this.numberOfDecisionClasses != 1) throw (new Exception("This is a regression method: wrong number of decision classes"));
		
		this.instances.insertAttributeAt(new Attribute("AuxiliaryDecision"), this.numberOfConditionAttributes + 1);
		this.auxiliaryDecisionAttribute = this.numberOfConditionAttributes + 1;
				
		this.instances.insertAttributeAt(new Attribute("InstanceIndex"), this.numberOfConditionAttributes + 2);
		this.instanceIndexAttribute = this.numberOfConditionAttributes + 2;
		
		for (int i = 0; i < this.numberOfInstances; i++) {
			this.instances.instance(i).setValue(this.instanceIndexAttribute, i);
			this.instances.instance(i).setValue(this.auxiliaryDecisionAttribute, this.instances.instance(i).classValue());
		}
		
		coveredInstances = new short[this.instances.numInstances()];
				
		this.ruleBuilder.initialize(this.instances);
		
		mainRandomGenerator = new Random();
	}
	
	public void updateFunction(double decision)  {
		for (int i = 0; i < this.instances.numInstances(); i++)
			if(this.coveredInstances[i] >= 0)
				this.valueOfF[i] += decision;
	}
	
	public double computeValueOfF(Instance instance) {
		ntb.input(instance); 
		instance = ntb.output(); 
		if (replaceMissingValues) {
			rmv.input(instance);
			instance = rmv.output();
		}
		double valueOfF = this.defaultRule;
		for (int i = 0; i < M; i++)
			valueOfF += rules[i].classifyInstance(instance);
		return valueOfF;
	}
	
	public double applyRule(int rule, Instance instance) {
		if(rule == 0) return this.defaultRule;
		ntb.input(instance); 
		instance = ntb.output();
		if (replaceMissingValues) {
			rmv.input(instance);
			instance = rmv.output();
		}
		return rules[rule - 1].classifyInstance(instance);
	}
	
	public double classifyInstance(Instance instance) {
		double[] distribution = null;
		try {
			distribution = this.distributionForInstance(instance);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return distribution[0];
	}
	
	public double [] distributionForInstance(Instance instance) throws Exception {
		//Specific for Regender
		if(this.instances.numClasses() != 1) throw (new Exception("This is a regression method: wrong number of decision classes"));
		double[] distribution = new double[instance.numClasses()];
		distribution[0] = this.computeValueOfF(instance);
		return distribution;
    }
	
	public String toString() {
		
		String string = new String();
		int i = 0;
		string = (0) + ". Default rule:\t" + this.defaultRule + "\n\n";
		while (i < M) {
			if (rules[i] != null) {
				string += (i + 1) + "." + rules[i].toString() + "\n";
				i++;
			}
			else {
				M = i;
			}
		}
		string += "\nFinished at " + M;
		return string;
		
	}
	
	public static void main(String[] args) {
		return;	
	}

	

	
	
	
		
}