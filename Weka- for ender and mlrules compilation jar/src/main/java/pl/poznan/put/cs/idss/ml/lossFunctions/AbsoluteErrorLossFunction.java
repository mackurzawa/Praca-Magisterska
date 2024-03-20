/*
 * Created on 2007-07-06
*/

package pl.poznan.put.cs.idss.ml.lossFunctions;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;

import pl.poznan.put.cs.idss.ml.empiricalRiskMinimizers.EmpiricalRiskMinimizer;
import weka.core.Instances;

/**
 * @author wk
 * @author kd
 */

public class AbsoluteErrorLossFunction extends LossFunction {

	public AbsoluteErrorLossFunction () {
		writeLog("Absolute Error!");
	}

	public double getLoss(double y, double f) {
		return Math.abs(y - f);
	}
	
	public double getFirstDerivative(double y, double f) {
		return -Math.signum(y - f); 
	}

	public double getSecondDerivative(double y, double f) {
		return 0; 
	}

	public double computeDistribution(double f) {
		if (f >= 0)
			return 1;
		else
			return 0;
	}
	
	public double[] computeDistribution(double[] f) {
		return null;
	}
	
	public double computeDecision(short[] coveredInstances, EmpiricalRiskMinimizer minimizer) {

		int numCovered = 0;
		for (int i = 0; i < coveredInstances.length; i++)
			if (coveredInstances[i] >= 0)
				numCovered++;

		writeLog(String.valueOf(numCovered));
			
		double[] values = new double[numCovered];
		double[] valueOfF = minimizer.getValueOfF();
		
		int index = 0;
		for (int i = 0; i < coveredInstances.length; i++)
			if (coveredInstances[i] >= 0)
					values[index++] = minimizer.getInstances().instance(i).value(minimizer.getAuxiliaryDecisionAttribute()) - valueOfF[i];
					writeLog(String.valueOf(values[index-1]));
		
		Arrays.sort(values);
		short[] transformed = new short[values.length];

		for (int j=0;j<values.length;j++) {
			transformed[j] = (short)values[j];
}
		writeLogArray(transformed);
		// writeLogArray(values);
		if (numCovered % 2 == 1)
			return values[numCovered / 2];
		else
			return (values[numCovered / 2] + values[numCovered / 2 - 1]) / 2;
	}

	public double computeDefaultDecision(short[] coveredInstances, EmpiricalRiskMinimizer minimizer) {
		// double a = this.computeDecision(coveredInstances, minimizer);
		// writeLog(String.valueOf(a));
		// return a;


		return this.computeDecision(coveredInstances, minimizer);
	}

	
	
	public static void main(String[] args) {
	}
}
