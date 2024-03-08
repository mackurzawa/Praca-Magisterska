/*
 * Created on 2007-07-05
 * 
 */

package pl.poznan.put.cs.idss.ml.ender;

import java.io.FileWriter;
import java.io.IOException;
import java.io.Serializable;

import pl.poznan.put.cs.idss.ml.empiricalRiskMinimizers.EmpiricalRiskMinimizer;

import pl.poznan.put.cs.idss.ml.lossFunctions.LossFunction;

import weka.core.Instances;
import weka.core.Utils;

/**
 * @author wk
 * @author kd
 */

public abstract class RuleBuilder implements Serializable {
	
	void writeLog(String s){
		try{
			FileWriter writer = new FileWriter("C:\\Users\\Maciej\\Desktop\\Praca Magisterska\\pila.txt", true);
			writer.write("RuleBuilder:\n");
			writer.write(s);
			writer.write("\n...\n");
			writer.close();
		} catch(IOException e){}
	}
	void writeLogArray2D(int[][] s){
		try{
			FileWriter writer = new FileWriter("C:\\Users\\Maciej\\Desktop\\Praca Magisterska\\pila.txt", true);
			for (int[] row : s) {
				for (int value : row) {
					writer.write(String.valueOf(value));
					writer.write(" ");
				}
				writer.write("\n");
			}
			writer.write("\n...\n");
			writer.close();
		} catch(IOException e){
	
		}
	}
	
	// inverted lists
	protected int[][] invertedList = null; 

	// ender
	protected Instances instances = null;
	
	protected EmpiricalRiskMinimizer empiricalLossMinimizer;

	protected LossFunction lossFunction;
	
	// precision 
	final static double EPSILON = 1e-8;

	protected int numberOfConditionAttributes;
	
	protected int instanceIndexAttribute;
	
	protected int auxiliaryDecisionAttribute = 0;

	protected class Cut {

		public double decision = 0;
		public int position = -1;
		public int direction = 0;
		public double value = 0;
		public double empiricalRisk = 0;
		public boolean exists = false;
		//public boolean emptyRule = false;
	
		Cut(){
			super();
		}
		
		Cut(Cut cut){
			this.copy(cut);
		}
		
		void copy(Cut cut){
			this.decision = cut.decision;
			this.position = cut.position;
			this.direction = cut.direction;
			this.value = cut.value;
			this.exists = cut.exists;
			this.empiricalRisk = cut.empiricalRisk;
		}
	}

	public double nu = 1;
	
	public void setNu(double nu) {
		this.nu = nu;
	}

	public double getNu() {
		return this.nu;
	}

	
	public RuleBuilder() {
		super();
	}

	public RuleBuilder(EmpiricalRiskMinimizer empiricalLossMinimizer) {
		super();
		this.empiricalLossMinimizer = empiricalLossMinimizer;
	}

	public EmpiricalRiskMinimizer getEmpiricalLossMinimizer() {
		return this.empiricalLossMinimizer;
	}
	
	public int[] getInvertedList(int attribute) {
		return this.invertedList[attribute];
	}
	
	public int getInvertedListValue(int attribute, int position) {
		return this.invertedList[attribute][position];
	}
		
	public void initialize(Instances instances) {
		
		this.instances = instances;

		this.numberOfConditionAttributes = this.instances.numAttributes() - 3; //two auxiliary attributes
		
		// writeLog("auxiliaryDecisionAttribute");
		// writeLog(String.valueOf(auxiliaryDecisionAttribute));
		// writeLog("numberOfConditionAttributes");
		// writeLog(String.valueOf(this.numberOfConditionAttributes));

		this.instanceIndexAttribute = this.numberOfConditionAttributes + 2;
		this.auxiliaryDecisionAttribute = this.numberOfConditionAttributes + 1;
		this.invertedList = new int[this.numberOfConditionAttributes][this.instances.numInstances()];
				
		for(int i = 0; i < this.numberOfConditionAttributes; i++) {
				this.instances.sort(i);
				double[] temporaryDoubleArray = this.instances.attributeToDoubleArray(this.instanceIndexAttribute);
				for(int j = 0; j < this.instances.numInstances(); j++) {
					invertedList[i][j] = (int) temporaryDoubleArray[j]; 
				}
		}
		// writeLogArray2D(invertedList);
		
		this.instances.sort(this.instanceIndexAttribute);
		
		/*for(int i = 0; i < this.numberOfConditionAttributes; i++) {
			System.out.print("Attribute: " + i + "\t");
			for(int j = 0; j < this.instances.numInstances(); j++) {
				System.out.print(Utils.doubleToString(this.instances.instance(invertedList[i][j]).value(i),4,2) + "\t");
			}
			System.out.println();
		}*/
		
		this.empiricalLossMinimizer.initialize(this.instances, this.auxiliaryDecisionAttribute);
	}

	abstract public Rule createRule(double[] valueOfF, short[] coveredInstances);

	abstract public double createDefaultRule(double[] valueOfF, short[] coveredInstances);
	
	}
