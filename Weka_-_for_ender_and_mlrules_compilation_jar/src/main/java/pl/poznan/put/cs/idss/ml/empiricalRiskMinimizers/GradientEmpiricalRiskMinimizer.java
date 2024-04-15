/*
 * Created on 2007-07-06
 *
 */
package pl.poznan.put.cs.idss.ml.empiricalRiskMinimizers;

import java.io.FileWriter;
import java.io.IOException;

import weka.core.Instances;

/**
 * @author wk
 * @author kd
 */

public class GradientEmpiricalRiskMinimizer extends EmpiricalRiskMinimizer {
	void writeLog(String s){
		try{
			FileWriter writer = new FileWriter("C:\\Users\\Maciej\\Desktop\\Praca Magisterska\\pila.txt", true);
			writer.write(s);
			writer.write("\n...\n");
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

	double[] gradient = null;
	
	public void initialize(Instances instances, int auxiliaryDecisionAttribute) {
		super.initialize(instances,auxiliaryDecisionAttribute);
		// writeLog("GradientEmpiricalRiskMinimizer");
		this.gradient = new double[this.instances.numInstances()];
	}
		
	double sumOfWeights = 0.0;
	int count = 0;
	
	public void initializeForCut() {
		this.sumOfWeights = 0.0;
		count = 0;
	}
		
	public void initializeForRule(double [] valueOfF, short[] coveredInstances) {
		super.setValueOfF(valueOfF);
		for (int i = 0; i < coveredInstances.length; i++) {
			if(coveredInstances[i] > 0) {
				this.gradient[i] = this.lossFunction.getFirstDerivative(this.instances.instance(i).value(this.auxiliaryDecisionAttribute), this.valueOfF[i]);
				// writeLog("this.instances.instance(i).value(this.auxiliaryDecisionAttribute))");
				// writeLog(String.valueOf(this.instances.instance(i).value(this.auxiliaryDecisionAttribute)));
				// writeLog("this.valueOfF[i]");
				// writeLog(String.valueOf(this.valueOfF[i]));
				// writeLog("this.gradient[i]");
				// writeLog(String.valueOf(this.gradient[i]));
			}	
		}
		// writeLog("this.auxiliaryDecisionAttribute");
		// writeLog(String.valueOf(auxiliaryDecisionAttribute));
		// writeLog("this.gradient just after calculating");
		// writeLogArray(this.gradient);
		//return 0; //- distance can not be greaten than zero
	}
	
	public double computeCurrentEmpiricalRisk (int position, int weight) {
		this.sumOfWeights += weight*this.gradient[position];
		this.count += weight;
		// writeLog("this.gradient");
		// writeLogArray(this.gradient);
		// writeLog("weight*this.gradient[position]:");
		// writeLog(String.valueOf(weight*this.gradient[position]));
		// writeLog("Sum of weights:");
		// writeLog(String.valueOf(sumOfWeights));
		// writeLog("count:");
		// writeLog(String.valueOf(count));
		return - Math.abs(this.sumOfWeights)/Math.sqrt(count);
	}
	

	public static void main(String[] args) {
	}
}
