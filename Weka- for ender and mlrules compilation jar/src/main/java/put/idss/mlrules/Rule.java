/*
 * Created on 2007-07-05
 * 
 */

package put.idss.mlrules;

import java.io.Serializable;
import java.util.Vector;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Utils;

/**
 * @author wk
 * @author kd
 */

public class Rule implements Serializable {
	
	static final long serialVersionUID = -1;
	
	final static byte GREATER_EQUAL = 1;
	final static byte LESS_EQUAL = -1;
	
	final static double MINUS_INFINITY = -1e40;
	final static double PLUS_INFINITY = 1e40;
	
	double[] decision;
	
	public Vector<double[]> selectors = new Vector<double[]>();
	Vector<Attribute> attributes = new Vector<Attribute>();
	
	private Attribute classAttribute = null;
	
	public Rule(Attribute classAttribute) {
		this.classAttribute = classAttribute;
	}
	
	
	public Rule(double attributeIndex, double cutValue, double direction, Attribute attribute, double[] decision, Attribute classAttribute) {
		addSelector(attributeIndex, cutValue, direction, attribute);
		setDecision(decision);
		this.classAttribute = classAttribute;
	}
	
	public void addSelector(double attributeIndex, double cutValue, double direction, Attribute attribute) {
		for (double[] selector : selectors)
			if (selector[0] == attributeIndex) {
				if ((short)direction == GREATER_EQUAL)
					selector[1] = Math.max(cutValue, selector[1]);
				else 
					selector[2] = Math.min(cutValue, selector[2]);
				return;
			}
		double[] selector = new double[3];
		selector[0] = attributeIndex;
		if ((short)direction == GREATER_EQUAL) {
			selector[1] = cutValue;
			selector[2] = PLUS_INFINITY;
		}
		else {
			selector[1] = MINUS_INFINITY;
			selector[2] = cutValue;
		}
		selectors.add(selector);
		attributes.add(attribute);
	}

	
	public void setDecision(double[] decision) {
		this.decision = decision;
	}
		
	public double[] getDecision() {
		return decision;
	}
	
	public double[] classifyInstance(Instance instance) {
		boolean covered = true;
		for (double[] selector : selectors) {
			if(instance.isMissing((int) selector[0]) == true) {
				covered = false;
				break;
			} 
			if (selector[1] > instance.value((int)selector[0]) || selector[2] < instance.value((int)selector[0])) {
				covered = false;
				break;
			}
		}
		if (covered == true)
			return decision;
		else
			return null;
	}
	
	public String toString() {
		StringBuffer string = new StringBuffer("Rule: \n");
		for (int i = 0; i < selectors.size(); i++) {
			double[] selector = (double[])selectors.get(i);
			String sign;
			if (attributes.get(i).isNominal()) {
				if (selector[1] == MINUS_INFINITY)
					sign = attributes.get(i).value(0);
				else
					sign = attributes.get(i).value(1);
				string.append("  " + attributes.get(i).name() + " is " + sign + "\n");
			}
			else {
				if (selector[1] == MINUS_INFINITY) 
					sign = " <= " + Utils.roundDouble(selector[2],3);
				else if (selector[2] == PLUS_INFINITY)
					sign = " >= " + Utils.roundDouble(selector[1],3);
				else
					sign = " in [" + Utils.roundDouble(selector[1],3) + "," + Utils.roundDouble(selector[2],3) + "]";
				string.append("  " + attributes.get(i).name() + sign + "\n");
			}
		}
		string.append("=> vote for class ");
		
		int i = 0;
		while (decision[i] < 0) i++;
		string.append(classAttribute.value(i) + " with weight " + decision[i] + "\n");
		return string.toString();
	}
	
}
