/*
 * Created on 2007-07-06
 *
 */

package pl.poznan.put.cs.idss.ml.empiricalRiskMinimizers;

import java.io.Serializable;

import pl.poznan.put.cs.idss.ml.lossFunctions.LossFunction;

import weka.core.Instances;

/**
 * @author wk
 * @author kd
 */

public abstract class EmpiricalRiskMinimizer implements Serializable {
	
	//public EmpiricalRiskMinimizer(double alpha) {
	//	this.alpha = alpha;
	//}
	
	Instances instances = null;
	
	int auxiliaryDecisionAttribute = -1;

	//alpha
	double alpha = 1.0;
	
	//values of F
	double[] valueOfF = null;
	
	final public void setValueOfF(double [] valueOfF) {
		this.valueOfF = valueOfF;
		//this.setAlpha(alpha);
	}
	
	public double[] getValueOfF() {
		return this.valueOfF;
	}
	
	public int getAuxiliaryDecisionAttribute() {
		return this.auxiliaryDecisionAttribute;
	}
	
	public Instances getInstances() {
		return this.instances;
	}
	
	//loss values
	//protected double[] loss; //??
	
	LossFunction lossFunction;
		
	public void setLossFunction(LossFunction lossFunction) {
		this.lossFunction = lossFunction;
	}
		
	public LossFunction getLossFunction() {
		return this.lossFunction;
	}
	
	public void setAlpha(double alpha) {
		this.alpha = alpha;
	}
	
	public double getAlpha() {
		return this.alpha;
	}
	
	
	public void initialize(Instances instances, int auxiliaryDecisionAttribute) {
		this.instances = instances;
		this.auxiliaryDecisionAttribute = auxiliaryDecisionAttribute;
		//this.loss = new double [this.instances.numInstances()];
	}
			
	abstract public void initializeForRule(double [] valueOfF, short[] coveredInstances);
	
	abstract public void initializeForCut();
		
	abstract public double computeCurrentEmpiricalRisk (int position, int weight);
	
	public double computeDecision(short[] coveredInstances) {
		return this.lossFunction.computeDecision(coveredInstances, this);
	}

	public double computeDefaultDecision(short[] coveredInstances) {
		return this.lossFunction.computeDefaultDecision(coveredInstances, this);
	}

	
	
	public double computeEmpiricalRisk (short[] coveredInstances) { //??
		double currentEmpiricalRisk = 0;
		for (int i = 0; i < coveredInstances.length; i++) {
			if(coveredInstances[i] > 0) {
				double loss = 
					this.lossFunction.getLoss(this.instances.instance(i).value(this.auxiliaryDecisionAttribute), valueOfF[i]);
				currentEmpiricalRisk += coveredInstances[i] * this.instances.instance(i).weight() * loss;
			}
		}	
		return currentEmpiricalRisk;
	}
	
}
