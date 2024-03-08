/*
 * Created on 2007-07-06
 *
 */
package pl.poznan.put.cs.idss.ml.lossFunctions;

import pl.poznan.put.cs.idss.ml.empiricalRiskMinimizers.EmpiricalRiskMinimizer;
import weka.core.Instances;

/**
 * @author wk
 * @author kd
 */

public class SquaredLossFunction extends LossFunction {

	double HIGH_VALUE = 1e12;
	
	public SquaredLossFunction () {
	}

	public double getLoss(double y, double f) {
		return Math.pow(y - f,2);
	}
	
	public double getFirstDerivative(double y, double f) {
		return -2*(y - f); 
	}

	public double getSecondDerivative(double y, double f) {
		return 2; 
	}

	public double computeDistribution(double f) {
		return Math.max(Math.min(1,0.5*(f + 1)),0);
	}
	
	public double[] computeDistribution(double[] f) {
		/*double totalF = 0;
		for (int i = 0; i < f.length; i++)
			totalF += f[i];
		totalF /= f.length;
		double[] distribution = new double[f.length];
		double total = 0;
		for (int i = 0; i < distribution.length; i++) {
			if (f[i] - totalF >= 1)
				distribution[i] = HIGH_VALUE;
			else
				distribution[i] = 1 / (1 - f[i] + totalF);
			total += distribution[i];
		}
		for (int i = 0; i < distribution.length; i++)
			distribution[i] /= total;
		return distribution;
		*/
		return null;
	}
	
	public double computeDecision(short[] coveredInstances, EmpiricalRiskMinimizer minimizer) {
		double decision = 0;
		double norm = 0;
		double[] valueOfF = minimizer.getValueOfF();
		
		for (int i = 0; i < coveredInstances.length; i++)
			if (coveredInstances[i] >= 0) {
				decision += /*coveredInstances[i] * */(minimizer.getInstances().instance(i).value(minimizer.getAuxiliaryDecisionAttribute()) - valueOfF[i]);
				norm += 1;//coveredInstances[i];
			}
			
		/*
		double currentEmpiricalRisk = 0;	
		for (int i = 0; i < coveredInstances.length; i++) {
			if(coveredInstances[i] >= 0) {
				currentEmpiricalRisk += Math.pow(minimizer.getInstances().instance(i).value(minimizer.getAuxiliaryDecisionAttribute()) - (valueOfF[i] + decision/norm), 2);
				//currentEmpiricalRisk += coveredInstances[i] * Math.pow(getFirstDerivative(this.instances.instance(i).value(this.auxiliaryDecisionAttribute), valueOfF[i]),2);
				
			}
		}
		System.out.println("After: " + currentEmpiricalRisk);
		*/
		// writeLog("value of f");
		// writeLogArray((short)valueOfF);
		writeLog("covered Instances");
		writeLogArray(coveredInstances);
		writeLog("decision");
		writeLog(String.valueOf(decision));
		writeLog("norm");
		writeLog(String.valueOf(norm));
			
		return decision / norm;
	}

	public double computeDefaultDecision(short[] coveredInstances, EmpiricalRiskMinimizer minimizer) {
		return this.computeDecision(coveredInstances, minimizer);
	}
	
	
	public static void main(String[] args) {
	}
}
