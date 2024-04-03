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

public class LeastAngleEmpiricalRiskMinimizer extends EmpiricalRiskMinimizer {
	void writeLog(String s){
		try{
			FileWriter writer = new FileWriter("C:\\Users\\Maciej\\Desktop\\Praca Magisterska\\pila.txt", true);
			writer.write("LeastAngleEmpiricalRiskMinimizer:\n");
			writer.write(s);
			writer.write("\n...\n");
			writer.close();
		} catch(IOException e){}
	}

	double[] gradient = null;
	
	public void initialize(Instances instances, int auxiliaryDecisionAttribute) {
		super.initialize(instances,auxiliaryDecisionAttribute);
		// writeLog("LeastAngleEmpiricalRiskMinimizer");
		this.gradient = new double[this.instances.numInstances()];
	}
		
	double sumOfWeights = 0.0;
	
	public void initializeForCut() {
		this.sumOfWeights = 0.0;
	}
		
	public void initializeForRule(double [] valueOfF, short[] coveredInstances) {
		super.setValueOfF(valueOfF);
		for (int i = 0; i < coveredInstances.length; i++) {
			if(coveredInstances[i] > 0) {
				this.gradient[i] = this.lossFunction.getFirstDerivative(this.instances.instance(i).value(this.auxiliaryDecisionAttribute), this.valueOfF[i]);
			}	
		}
	}
	
	public double computeCurrentEmpiricalRisk (int position, int weight) {
		this.sumOfWeights += weight*this.gradient[position];
		return - Math.abs(this.sumOfWeights);
	}
	
	public static void main(String[] args) {
	}
}
