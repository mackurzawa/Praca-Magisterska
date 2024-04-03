/*
 * Created on 2007-07-05
 * 
 */

package put.idss.mlrules;

import weka.core.logging.ConsoleLogger;
import weka.core.logging.Logger;
import weka.core.logging.Logger.Level;

import java.io.FileWriter;
import java.io.IOException;
import java.io.Serializable;
import java.lang.reflect.Array;
import java.util.Arrays;

import weka.core.Instances;


/**
 * @author Wojciech Kotlowski
 * @author Krzysztof Dembczynski
 */




public class RuleBuilder implements Serializable {

	// public String filename = "C:\\Users\\macku\\Desktop\\Praca Magisterska\\Praca-Magisterska\\pila.txt";
	public String filename = "C:\\Users\\Maciej\\Desktop\\Praca Magisterska\\pila.txt";
	
	void writeLog(String s){
		try{
			FileWriter writer = new FileWriter(filename, true);
			// for (int[] row : invertedList) {
			// 	for (int value : row) {
			// 		writer.write(String.valueOf(value));
			// 		writer.write(" ");
			// 	}
			// 	writer.write("\n");
			// }
			writer.write(s);
			writer.write("\n...\n");
			writer.close();
		} catch(IOException e){
	
		}
	}

	void writeLogArray(double[] s){
		try{
			FileWriter writer = new FileWriter(filename, true);
	
			for (double value : s) {
				writer.write(String.valueOf(value));
				writer.write(" ");
			}
			writer.write("\n...\n");
			writer.close();
		} catch(IOException e){
	
		}
	}

	void writeLogArray2D(double[][] s){
		try{
			FileWriter writer = new FileWriter(filename, true);
			for (double[] row : s) {
				for (double value : row) {
					writer.write(String.valueOf(value));
					writer.write(" ");
				}
				writer.write("\n");
			}
			writer.write("\n...\n");
			writer.close();
		} catch(IOException e){
	
		}
	}

	static final long serialVersionUID = -1;

	// precision 
	final static double EPSILON = 1e-8;
	
	// regularization constant
	public double R = 10.0;
	
	// regularization constant per probability
	public double Rp = 1e-2;

	// inverted lists
	protected int[][] invertedList = null; 

	protected Instances instances = null;
	
	protected int D;
	
	protected int indexAttribute;
	
	protected int K;
	
	protected int N;
	
	protected boolean useLineSearch = false;
	
	protected double nu = 0.1;
	
	protected double[][] f = null;
	
	private double[][] probability = null;
	
	private double gradient = 0;
	
	private double hessian = 0;
	
	private int maxK = 0;
	
	public boolean preChosenK = false;
	
	private double[] gradients = null;
	
	private double[] hessians = null;
	
	private boolean useGradient = true;
	
	private double lineSearchMax = 4.0;
	
	private int lineSearchIterations = 10;
	
	private double lineSearchPrecision = 1e-4;
	
	public RuleBuilder(double nu, boolean useLineSearch, boolean useGradient, boolean preChosenK, double R, double Rp) {
		this.nu = nu;
		this.useLineSearch = useLineSearch;
		this.useGradient = useGradient;
		this.preChosenK = preChosenK;
		this.R = R;
		this.Rp = Rp;
		// writeLog("nu" + String.valueOf(nu));
		// writeLog("useLineSearch" + String.valueOf(useLineSearch));
		// writeLog("useGradient" + String.valueOf(useGradient));
		// writeLog("preChosenK" + String.valueOf(preChosenK));
		// writeLog("R" + String.valueOf(R));
		// writeLog("Rp" + String.valueOf(Rp));
	}

	protected final class Cut {

		public double[] decision = null;
		public int position = -1;
		public int direction = 0;
		public double value = 0;
		public double empiricalRisk = 0;
		public boolean exists = false;
	
		Cut(int K){
			super();
			decision = new double[K];
		}
		
		Cut(Cut cut){
			decision = new double[cut.decision.length];
			copy(cut);
		}
		
		void copy(Cut cut){
			for (int i = 0; i < decision.length; i++)
				decision[i] = cut.decision[i];
			position = cut.position;
			direction = cut.direction;
			value = cut.value;
			exists = cut.exists;
			empiricalRisk = cut.empiricalRisk;
		}
		
		public void saveCut(int direction, double currentValue, double value, double tempEmpiricalRisk) {
			this.direction = direction;
			this.value = (currentValue + value)/2;
			this.empiricalRisk = tempEmpiricalRisk;
			this.exists = true;
		}
	}

	public void initialize(Instances instances) {
		
		this.instances = instances;
		N = instances.numInstances();
		D = instances.numAttributes() - 2;
		K = instances.numClasses();
		indexAttribute = D + 1;
				
		invertedList = new int[D][N];
		
		for(int j = 0; j < D; j++) {
			instances.sort(j);
			double[] temporaryDoubleArray = instances.attributeToDoubleArray(indexAttribute);
			for(int i = 0; i < N; i++) {
				invertedList[j][i] = (int)temporaryDoubleArray[i]; 
			}
		}
		
		instances.sort(indexAttribute);

		
		probability = new double[N][K];
		gradients = new double[K];
		hessians = new double[K];
	}
	
	private void initializeForCut() {
		gradient = 0;
		hessian = R;
		Arrays.fill(gradients, 0);
		Arrays.fill(hessians, R);
	}
				
	public void initializeForRule(double[][] f, short[] coveredInstances) {
		this.f = f;
		if (preChosenK) {
			Arrays.fill(gradients, 0.0);
			Arrays.fill(hessians, R);
		}
		
		for (int i = 0; i < N; i++)
			if(coveredInstances[i] >= 0) {
				double norm = 0;
				for (int k = 0; k < K; k++) {
					probability[i][k] = Math.exp(f[i][k]);
					norm += probability[i][k];
				}
				for (int k = 0; k < K; k++) {
					probability[i][k] /= norm;
					if (preChosenK) {
						gradients[k] -= instances.instance(i).weight() * probability[i][k];
						hessians[k] += instances.instance(i).weight() * (Rp + probability[i][k] * (1 - probability[i][k]));
					}
				}
				// writeLog("temp gradients");
				// writeLogArray(gradients);
				if (preChosenK)
					// writeLog("y:");
					// writeLog(String.valueOf((int)instances.instance(i).classValue()));
					gradients[(int)instances.instance(i).classValue()] += instances.instance(i).weight();
			}
		
		if (preChosenK) {
			maxK = 0;
			if (useGradient) {
				// writeLog("Gradients:");
				// writeLogArray(gradients);
				for (int k = 1; k < K; k++)
					if (gradients[k] > gradients[maxK])
						maxK = k;
			}
			else {
				for (int k = 1; k < K; k++)
					if (gradients[k] / Math.sqrt(hessians[k]) > gradients[maxK] / Math.sqrt(hessians[maxK]))
						maxK = k;
			}
		}
	}
		
	public double computeCurrentEmpiricalRisk(int position, int weight) {
		
		if (preChosenK) {
			if ((int)instances.instance(position).classValue() == maxK) 
				gradient += instances.instance(position).weight() * weight;
			gradient -= instances.instance(position).weight() * weight * probability[position][maxK];
			// writeLog("gradient: " + String.valueOf(gradient));
			if (useGradient)
				return -gradient;
			else {
				hessian += instances.instance(position).weight() * weight * (Rp + probability[position][maxK] * (1 - probability[position][maxK]));
				return - gradient * Math.abs(gradient) / hessian;
			}
		}
		else {
			int y = (int)instances.instance(position).classValue();
			for (int k = 0; k < K; k++) {
				if (y == k)
					gradients[k] += instances.instance(position).weight() * weight;
				gradients[k] -= instances.instance(position).weight() * weight * probability[position][k];
				if (!useGradient) {
					hessians[k] += instances.instance(position).weight() * weight * (Rp + probability[position][k] * (1 - probability[position][k]));
				}
			}
			
			if (useGradient) {
				double highest = gradients[0];
				for (int k = 1; k < K; k++)
					highest = Math.max(highest, gradients[k]);
				return - highest;
			}
			else {
				double highest = gradients[0] * Math.abs(gradients[0]) / hessians[0];
				for (int k = 1; k < K; k++)
					highest = Math.max(highest, gradients[k] * Math.abs(gradients[k]) / hessians[k]);
				return - highest;				
			}
		}
	}
	
	public double[] computeDecision(short[] coveredInstances) {

		if (preChosenK) {
			hessian = R;
			gradient = 0;
			writeLog("halko prechosenK wybrane");

			writeLog("probability");
			writeLogArray2D(probability);

			double[] transformed = new double[N];
			for (int j=0;j<N;j++) {
				transformed[j] = (double)coveredInstances[j];
			}
			// writeLog("covered Instances:");
			// writeLogArray(transformed);
		
			for (int i = 0; i < coveredInstances.length; i++){
				if (coveredInstances[i] >= 0) {
					if ((int)this.instances.instance(i).classValue() == maxK)
						gradient += instances.instance(i).weight();
					gradient -= instances.instance(i).weight() * probability[i][maxK];
					hessian += instances.instance(i).weight() * (Rp + probability[i][maxK] * (1 - probability[i][maxK]));
					if (instances.instance(i).weight() != 1){
						writeLog("weights");
						writeLog(String.valueOf(instances.instance(i).weight()));	
					}
				}
			}

			writeLog(String.valueOf(hessian));
			writeLog(String.valueOf(gradient));
		
			if (gradient <= 0)
				return null;

			writeLog("Max_K:");
			writeLog(String.valueOf(maxK));
			
			double alphaNR = gradient / hessian;
			writeLog("alphanr:");
			writeLog(String.valueOf(alphaNR));
			double[] decision = new double[K];
			Arrays.fill(decision, - alphaNR / K);
			decision[maxK] = alphaNR * (K - 1) / K;
			return decision;
		}
		else {
			writeLog("halko prechosenK NIE wybrane");
				Arrays.fill(hessians, R);
			Arrays.fill(gradients, 0);
			int chosenK = 0;
			double[] origGradients = new double[K];
		
			for (int i = 0; i < coveredInstances.length; i++)
				if (coveredInstances[i] >= 0) {
					for (int k = 0; k < K; k++) {
						if ((int)this.instances.instance(i).classValue() == k) {
							gradients[k] += instances.instance(i).weight();
							origGradients[k] += instances.instance(i).weight() * coveredInstances[i];
						}
						gradients[k] -= instances.instance(i).weight() * probability[i][k];
						hessians[k] += instances.instance(i).weight() * (Rp + probability[i][k] * (1 - probability[i][k]));
						origGradients[k] -= instances.instance(i).weight() * coveredInstances[i] * probability[i][k];
					}
				}
			// writeLogArray(gradients);
			// writeLogArray(hessians);
			// writeLogArray(origGradients);
		
			for (int k = 1; k < K; k++)
				if (origGradients[k] > origGradients[chosenK])
					chosenK = k;
			
			if (gradients[chosenK] <= 0)
				return null;
			
			double alphaNR = gradients[chosenK] / hessians[chosenK];
			double[] decision = new double[K];
			Arrays.fill(decision, - alphaNR / K);
			decision[chosenK] = alphaNR * (K - 1) / K;
			return decision;
		}
	}
	
	public Cut findBestCut(int attribute, short[] coveredInstances) {	
		Cut bestCut = new Cut(K);
		bestCut.position = -1; 
		bestCut.exists = false;
		bestCut.empiricalRisk = 0;
			
		double tempEmpiricalRisk = 0;

		for(int cutDirection = -1; cutDirection <= 1; cutDirection += 2) {			
			initializeForCut();
		
			int currentPosition = 0;
			int i;
			i = (cutDirection == Rule.GREATER_EQUAL)? N - 1 : 0;
			while((cutDirection == Rule.GREATER_EQUAL && i >= 0) || (cutDirection != Rule.GREATER_EQUAL && i < N)) {
				currentPosition = invertedList[attribute][i];
				if(coveredInstances[currentPosition] > 0 && instances.instance(currentPosition).isMissing(attribute) == false)
					break;
				if(cutDirection == Rule.GREATER_EQUAL) 
					i--; 
				else i++;
			}
		
			double currentValue = instances.instance(currentPosition).value(attribute);
			// writeLog("...");
			// writeLog("...");
			// writeLog("current_value");
			// writeLog(String.valueOf(currentValue));
			// writeLog("current position");
			// writeLog(String.valueOf(currentPosition));
			// writeLog("i:");
			// writeLog(String.valueOf(i));
			while((cutDirection == Rule.GREATER_EQUAL && i >= 0) || (cutDirection != Rule.GREATER_EQUAL && i < instances.numInstances())) {
				int nextPosition = invertedList[attribute][i];

				// writeLog("next position: " + String.valueOf(nextPosition));
				
				if(coveredInstances[nextPosition] > 0) {
					if(instances.instance(nextPosition).isMissing(attribute) == false) {
						double value = instances.instance(nextPosition).value(attribute);
						// writeLog("value: " + String.valueOf(value));
						if (currentValue != value) 
							if (tempEmpiricalRisk < bestCut.empiricalRisk + RuleBuilder.EPSILON)
								bestCut.saveCut(cutDirection, currentValue, value, tempEmpiricalRisk);
						tempEmpiricalRisk = computeCurrentEmpiricalRisk(nextPosition, coveredInstances[nextPosition]);
						// writeLog("temp empirical risk" + String.valueOf(tempEmpiricalRisk));
						currentValue = instances.instance(nextPosition).value(attribute);
					}
				}
				if(cutDirection == Rule.GREATER_EQUAL) i--; else i++;
			}
			// writeLog("End of calculating");
		}
		// writeLog("decisiom:");
		// writeLogArray(bestCut.decision);
		// writeLog("position:");
		// writeLog(String.valueOf(bestCut.position));
		// writeLog("direction:");
		// writeLog(String.valueOf(bestCut.direction));
		// writeLog("value:");
		// writeLog(String.valueOf(bestCut.value));
		// writeLog("empirical_risk:");
		// writeLog(String.valueOf(bestCut.empiricalRisk));
		// writeLog("exists:");
		// writeLog(String.valueOf(bestCut.exists));

		return bestCut;
	}
	
	public short[] markCoveredInstances(int bestAttribute, short[] coveredInstances, Cut cut) { 
		for(int i = 0; i < N; i++) 
			if (coveredInstances[i] != -1) {
				if(instances.instance(i).isMissing(bestAttribute) == true)
					coveredInstances[i] = -1;
				else {
					double value = instances.instance(i).value(bestAttribute);
					if (((value < cut.value) && (cut.direction == 1)) ||
						((value > cut.value) && (cut.direction == -1)))
						coveredInstances[i] = -1;
				}
			}
		return coveredInstances;
	}
	
	private double computeRuleGradient(double[][] f, short[] coveredInstances, double point) {
		int size = 0;
		double gradient = 0;
		for (int i = 0; i < N; i++)
			if(coveredInstances[i] >= 0) {
				size++;
				probability[i][maxK] = Math.exp(f[i][maxK] + point);
				double norm = probability[i][maxK];
				for (int k = 0; k < K; k++)
					if (k != maxK) {
						probability[i][k] = Math.exp(f[i][k]);
						norm += probability[i][k];
					}
				if ((int)instances.instance(i).classValue() == maxK)
					gradient += instances.instance(i).weight();
				gradient -= instances.instance(i).weight() * probability[i][maxK] / norm;
			}
		return gradient / size;
	}
	
	public double[] getLineSearchDecision(double[][] f, short[] coveredInstances) {
		double gradient = computeRuleGradient(f, coveredInstances, lineSearchMax);
		if (gradient >= 0) {
			double[] decision = new double[K];
			Arrays.fill(decision, - lineSearchMax / K);
			decision[maxK] = lineSearchMax * (K - 1) / K;
			return decision;			
		}
		else
			return getLineSearchDecision(f, coveredInstances, 0, lineSearchMax, 1);
	}
	
	private double[] getLineSearchDecision(double[][] f, short[] coveredInstances, 
			double left, double right, int depth) {
		double middle = (left + right) / 2.0;
		double gradient = computeRuleGradient(f, coveredInstances, middle);		
		if (Math.abs(gradient) < lineSearchPrecision || depth == lineSearchIterations) {
			double[] decision = new double[K];
			Arrays.fill(decision, - middle / K);
			decision[maxK] = middle * (K - 1) / K;
			return decision;			
		}
		else {
			if (gradient > 0) 
				return getLineSearchDecision(f, coveredInstances, 
						middle, right, depth + 1);
			else
				return getLineSearchDecision(f, coveredInstances, 
						left, middle, depth + 1);
		}	
	}
	
	public Rule createRule(double[][] f, short[] coveredInstances) {
		
		initializeForRule(f, coveredInstances);

		Rule rule = new Rule(instances.classAttribute());

		Cut bestCut = new Cut(K);
		bestCut.empiricalRisk = 0;
		boolean creating = true;

		while(creating){
			int bestAttribute = -1;
			Cut cut;
			for (int j = 0; j < D; j++) {
				cut = findBestCut(j, coveredInstances);
				if (cut.empiricalRisk < bestCut.empiricalRisk - RuleBuilder.EPSILON) {
					bestCut.copy(cut);
					bestAttribute = j;
				}
			}			
			if(bestAttribute == -1 || bestCut.exists == false) {
				creating = false;
			}
			else {
				rule.addSelector(bestAttribute, bestCut.value, bestCut.direction, instances.attribute(bestAttribute));
				coveredInstances = markCoveredInstances(bestAttribute, coveredInstances, bestCut);
			}
		}
				
		if (bestCut.exists == true) {
			double[] decision = null;
			if (useLineSearch)
				decision = getLineSearchDecision(f, coveredInstances);
			else
				decision = computeDecision(coveredInstances);
			if (decision == null)
				return null;
			for (int i = 0; i < decision.length; i++)
				decision[i] *= nu;
			rule.setDecision(decision);
			return rule;
		}
		else 
			return null;
	}
	
	public double[] createDefaultRule(double[][] f, short[] coveredInstances) {
		initializeForRule(f, coveredInstances);
		double[] decision = computeDecision(coveredInstances);
		for (int i = 0; i < decision.length; i++)
			decision[i] *= nu;
		writeLog("decision:");
		writeLogArray(decision);
		return decision;
	}
	
	public double[] createDefaultRule() {
		double[] priors = new double[K];
		for (int i = 0; i < N; i++)
			priors[(int)instances.instance(i).classValue()]++;
		int emptyClasses = 0;
		for (int k = 0; k < K; k++) {
			priors[k] /= (double)N;
			if (priors[k] == 0)
				emptyClasses++;
		}
		double logPriors = 0;
		for (int k = 0; k < K; k++)
			if (priors[k] != 0)
				logPriors += Math.log(priors[k]);
		logPriors /= (K - emptyClasses);
		
		double[] decision = new double[K];
		Arrays.fill(decision, -logPriors);
		for (int k = 0; k < K; k++)
			if (priors[k] != 0)
				decision[k] += Math.log(priors[k]);
			else
				decision[k] = 0;
		return decision;
	}

}
