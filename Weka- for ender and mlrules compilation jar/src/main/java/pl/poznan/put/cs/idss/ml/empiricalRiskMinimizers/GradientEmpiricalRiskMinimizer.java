/*
 * Created on 2007-07-06
 *
 */
package pl.poznan.put.cs.idss.ml.empiricalRiskMinimizers;

import weka.core.Instances;

/**
 * @author wk
 * @author kd
 */

public class GradientEmpiricalRiskMinimizer extends EmpiricalRiskMinimizer {

	double[] gradient = null;
	
	public void initialize(Instances instances, int auxiliaryDecisionAttribute) {
		super.initialize(instances,auxiliaryDecisionAttribute);
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
			}	
		}
		//return 0; //- distance can not be greaten than zero
	}
	
	public double computeCurrentEmpiricalRisk (int position, int weight) {
		this.sumOfWeights += weight*this.gradient[position];
		this.count += weight;
		return - Math.abs(this.sumOfWeights)/Math.sqrt(count);
	}
	

	public static void main(String[] args) {
	}
}
