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

public class AbsoluteErrorRiskMinimizer extends EmpiricalRiskMinimizer {

	double[] gradient = null;
	
	public void initialize(Instances instances, int auxiliaryDecisionAttribute) {
		super.initialize(instances,auxiliaryDecisionAttribute);
		this.gradient = new double[this.instances.numInstances()];
	}
		
	double sumOfWeights = 0.0;
	double sumOfZeroResiduals = 0.0;
		
	public void initializeForRule(double [] valueOfF, short[] coveredInstances) {
		super.setValueOfF(valueOfF);
		for (int i = 0; i < coveredInstances.length; i++) {
			if(coveredInstances[i] > 0) {
				this.gradient[i] = this.lossFunction.getFirstDerivative(this.instances.instance(i).value(this.auxiliaryDecisionAttribute), this.valueOfF[i]);
			}	
		}
	}
	
	public void initializeForCut() {
		this.sumOfWeights = 0.0;
		this.sumOfZeroResiduals = 0.0;
	}

	
	public double computeCurrentEmpiricalRisk (int position, int weight) {
		this.sumOfWeights += weight*this.gradient[position];
		if(this.gradient[position] == 0.0) this.sumOfZeroResiduals += 1.0;
				
		return - Math.abs(this.sumOfWeights) + this.sumOfZeroResiduals;
	}
	
	
}
