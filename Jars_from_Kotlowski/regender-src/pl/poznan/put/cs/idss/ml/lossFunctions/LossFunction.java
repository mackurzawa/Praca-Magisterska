/*
 * Created on 2007-07-06
 *
 */

package pl.poznan.put.cs.idss.ml.lossFunctions;

import java.io.Serializable;

import pl.poznan.put.cs.idss.ml.empiricalRiskMinimizers.EmpiricalRiskMinimizer;

/**
 * @author wk
 * @author kd
 */

public abstract class LossFunction implements Serializable {

	public abstract double getLoss(double y, double f);
	
	public abstract double getFirstDerivative(double y, double f);

	public abstract double getSecondDerivative(double y, double f);
	
	public abstract double computeDistribution(double f);
	
	public abstract double[] computeDistribution(double[] f);
	
	public abstract double computeDecision(short[] coveredInstances, EmpiricalRiskMinimizer minimizer);
	
	public abstract double computeDefaultDecision(short[] coveredInstances, EmpiricalRiskMinimizer minimizer);
}
