/*
 * Created on 2007-07-05
 */

package pl.poznan.put.cs.idss.ml.ender;

import pl.poznan.put.cs.idss.ml.empiricalRiskMinimizers.EmpiricalRiskMinimizer;
import weka.core.Instances;


/**
 * @author wk
 * @author kd
 */

public class SingleRuleBuilder extends RuleBuilder {
	
	public SingleRuleBuilder() {
		super();
	}
	
	public SingleRuleBuilder(EmpiricalRiskMinimizer lossFunction) {
		super(lossFunction);
	}
	
	
	public void initialize(Instances instances) {
		super.initialize(instances);		
	}

	protected void saveCut(Cut cut, int direction, double currentValue, double value, double tempEmpiricalRisk) {
		cut.direction = direction;
		cut.value = (currentValue + value)/2;
		cut.empiricalRisk = tempEmpiricalRisk;
		cut.exists = true;
	}
	
	public Cut findBestCut(int attribute, short[] coveredInstances) {	
		//initialize data structures
		Cut bestCut = new Cut();
		bestCut.position = -1; 
		bestCut.exists = false;
		//bestCut.emptyRule = false;
		bestCut.empiricalRisk = 0;
			
		double tempEmpiricalRisk = 0;

		for(int cutDirection = -1; cutDirection <= 1; cutDirection += 2) {
			this.empiricalLossMinimizer.initializeForCut();		
			
			int currentPosition = 0;
			int i;
			i = (cutDirection == Rule.GREATER_EQUAL)? this.instances.numInstances() - 1 : 0;
			while((cutDirection == Rule.GREATER_EQUAL && i >= 0) || (cutDirection != Rule.GREATER_EQUAL && i < this.instances.numInstances())) {
				currentPosition = this.getInvertedListValue(attribute, i);
				if(coveredInstances[currentPosition] > 0 && this.instances.instance(currentPosition).isMissing(attribute) == false) {
					break;
				}
				if(cutDirection == Rule.GREATER_EQUAL) i--; else i++;
			}
		
			double currentValue = this.instances.instance(currentPosition).value(attribute);
			int count = 0;
			while((cutDirection == Rule.GREATER_EQUAL && i >= 0) || (cutDirection != Rule.GREATER_EQUAL && i < this.instances.numInstances())) {
				int nextPosition = this.getInvertedListValue(attribute, i);
				if(coveredInstances[nextPosition] > 0) {
					if(this.instances.instance(nextPosition).isMissing(attribute) == false) {
						count++;
						double value = this.instances.instance(nextPosition).value(attribute);
						int weight = (int) this.instances.instance(nextPosition).weight();
						if (currentValue != value && count >= 10) 
							if (tempEmpiricalRisk < bestCut.empiricalRisk - RuleBuilder.EPSILON)
								this.saveCut(bestCut, cutDirection, currentValue, value, tempEmpiricalRisk);
							//compute values for the next cut 
						tempEmpiricalRisk = this.empiricalLossMinimizer.computeCurrentEmpiricalRisk(nextPosition, coveredInstances[nextPosition]*weight);
						//System.out.println(tempEmpiricalRisk);
						currentValue = this.instances.instance(nextPosition).value(attribute);
					}
				}
				if(cutDirection == Rule.GREATER_EQUAL) i--; else i++;
			}
		}

		return bestCut;
	}
		

	public short[] markCoveredInstances(int bestAttribute, short[] coveredInstances, Cut cut) { 
		
		for(int i = 0; i < this.instances.numInstances(); i++) {
			if(this.instances.instance(i).isMissing(bestAttribute) == true) {
				coveredInstances[i] = -1;
			} else {
				double value = this.instances.instance(i).value(bestAttribute);
				if (((value < cut.value) && (cut.direction == 1)) ||
					((value > cut.value) && (cut.direction == -1))) {
					coveredInstances[i] = -1;
				}
			}
		}
		
		return coveredInstances;
	}

	public Rule createRule(double[] valueOfF, short[] coveredInstances) {
		
		//compute current value of empirical risk
		this.empiricalLossMinimizer.initializeForRule(valueOfF, coveredInstances);
		//double currentEmpiricalRisk = this.empiricalLossMinimizer.computeEmpiricalRisk(coveredInstances);
		
		Rule rule = new Rule();
		Cut bestCut = new Cut();
		bestCut.empiricalRisk = 0; //currentEmpiricalRisk;
		boolean creating = true;
		
		//System.out.println(bestCut.empiricalRisk);

		int countOfCuts = 0;
		while(creating){
			int bestAttribute = -1;
			Cut cut;
			for (int i = 0; i < this.numberOfConditionAttributes; i++) {
				//start with empty rule; all instances are out of the rule; check only covered instances
				cut = this.findBestCut(i, coveredInstances);
				if (cut.empiricalRisk < bestCut.empiricalRisk - RuleBuilder.EPSILON) {
					bestCut.copy(cut);
					bestAttribute = i;
				}
			}
			// no cut was found or it is trivial
			
			if(bestAttribute == -1 || bestCut.exists == false) {// || bestCut.emptyRule == true) {
				creating = false;
			}
			else {
				//extend rule by additional cut and set decision			
				rule.addCondition(bestAttribute, bestCut.value, bestCut.direction, this.instances.attribute(bestAttribute).name());
				
				// filter all the instances; delete uncovered instances
				coveredInstances = this.markCoveredInstances(bestAttribute, coveredInstances, bestCut);

				//countOfCuts++;
				//if(countOfCuts > 10) creating = false;
			}
		}
		
		//System.out.println(bestCut.empiricalRisk);
				
		if (bestCut.exists == true) {// && bestCut.emptyRule == false) { 
			//compute decision
			double decision = this.nu * this.empiricalLossMinimizer.computeDecision(coveredInstances);
			rule.setDecision(decision);
			
			int numCoveredInstances = 0;
			int positive = 0;
			int negative = 0;
			for(int i = 0; i < coveredInstances.length; i++) {
				if(coveredInstances[i] >= 0) {
					numCoveredInstances++;
					if(instances.instance(i).classValue() > 0.5)
						positive++;
					else
						negative++;
				}
				
			}
			
			rule.setNumCoveredInstances(numCoveredInstances);
			rule.setNumCoveredPositiveInstances(positive);
			rule.setNumCoveredNegativeInstances(negative);
			return rule;
		}
		else { 
			return null;
		}
	}
	
	public double createDefaultRule(double[] valueOfF, short[] coveredInstances) {
		this.empiricalLossMinimizer.setValueOfF(valueOfF);
		return this.empiricalLossMinimizer.computeDefaultDecision(coveredInstances);
	}
	

}
