/*
 * Created on 2007-07-05
 * 
 */

package pl.poznan.put.cs.idss.ml.ender;

import java.io.Serializable;
import java.util.Vector;

import weka.core.Instance;

/**
 * @author wk
 * @author kd
 */

public class Rule implements Serializable {
	
	final static byte GREATER_EQUAL = 1;
	final static byte LESS_EQUAL = -1;
	
	final static double MINUS_INFINITY = -1e40;
	final static double PLUS_INFINITY = 1e40;

	public Vector<double[]> conditions = new Vector<double[]>();
	Vector<String> attributeNames = new Vector<String>();
	
	//rule output
	double decision;
	
	//rule properties
	private int numCoveredInstances = 0;
	private int numCoveredPositiveInstances = 0;
	private int numCoveredNegativeInstances = 0;
	private double absoluteDecision = 0.0;
	
	public Rule() {
	}
	
	public Rule(double decision) {
		this.decision = decision;
	}
	
	public Rule(double attributeIndex, double cutValue, double direction, String attributeName, double decision) {
		this.addCondition(attributeIndex, cutValue, direction, attributeName);
		this.setDecision(decision);
	}
	
	public void addCondition(double attributeIndex, double cutValue, double direction, String attributeName) {
		for (double[] condition : conditions)
			if (condition[0] == attributeIndex) {
				if ((short)direction == GREATER_EQUAL)
					condition[1] = Math.max(cutValue, condition[1]);
				else 
					condition[2] = Math.min(cutValue, condition[2]);
				return;
			}
		double[] condition = new double[3];
		condition[0] = attributeIndex;
		if ((short)direction == GREATER_EQUAL) {
			condition[1] = cutValue;
			condition[2] = PLUS_INFINITY;
		}
		else {
			condition[1] = MINUS_INFINITY;
			condition[2] = cutValue;
		}
		conditions.add(condition);
		attributeNames.add(attributeName);
	}

	
	public void setDecision(double decision) {
		this.decision = decision;
	}
	
		
	public double getDecision() {
		return decision;
	}
	
	public double classifyInstance(Instance instance) {

		boolean covered = true;
		for (double[] condition : conditions) {
			if(instance.isMissing((int) condition[0]) == true) {

				covered = false;
				break;
			}
			if (condition[1] > instance.value((int)condition[0]) || condition[2] < instance.value((int)condition[0])) {

				covered = false;
				break;
			}
		}
		if (covered == true)
			return decision;
		else
			return 0;
	}
	
	public String toString() {
		StringBuffer string = new StringBuffer("Rule: \n");
		for (int i = 0; i < conditions.size(); i++) {
			double[] condition = (double[])conditions.get(i);
			String sign;
			if (condition[1] == MINUS_INFINITY) 
				sign = " <= " + condition[2];
			else if (condition[2] == PLUS_INFINITY)
				sign = " >= " + condition[1];
			else
				sign = " in [" + condition[1] + "," + condition[2] + "]";
			
			string.append("  " + attributeNames.get(i) + sign + "\n");
		}
		string.append("=> Decision " + this.decision + "\n");
		string.append("Number of Covered Examples: " + this.numCoveredInstances +  "\n");
		string.append("Positive Examples: " + this.numCoveredPositiveInstances +  "\n");
		string.append("Negative Examples: " + this.numCoveredNegativeInstances +  "\n");

		return string.toString();
	}

	public void setNumCoveredInstances(int numCoveredInstances) {
		this.numCoveredInstances = numCoveredInstances;
	}
	
	public int getNumCoveredInstances() {
		return this.numCoveredInstances;
	}
	
	
	public void setNumCoveredPositiveInstances(int numCoveredPositiveInstances) {
		this.numCoveredPositiveInstances = numCoveredPositiveInstances;
		
	}


	public int getNumCoveredPositiveInstances() {
		return this.numCoveredPositiveInstances;
	}
	
	public void setNumCoveredNegativeInstances(int numCoveredNegativeInstances) {
		this.numCoveredNegativeInstances = numCoveredNegativeInstances;
		
	}
	
	public int getNumCoveredNegativeInstances() {
		return this.numCoveredNegativeInstances;
	}
	
	public void setAbsoluteDecision(double absoluteDecision) {
		this.absoluteDecision = absoluteDecision;
	}
	
	public double getAbsoluteDecision() {
		return this.absoluteDecision;
	}

	
	
}
