// /*
//  * Created on 2007-07-05
//  * 
//  */

// package put.idss.mlrules;

// import weka.core.logging.ConsoleLogger;
// import weka.core.logging.Logger;
// import weka.core.logging.Logger.Level;

// import java.io.FileWriter;
// import java.io.IOException;
// import java.io.Serializable;
// import java.lang.reflect.Array;
// import java.util.Arrays;

// import weka.core.Instances;


// /**
//  * @author Wojciech Kotlowski
//  * @author Krzysztof Dembczynski
//  */




// public class RuleBuilder implements Serializable {

// 	// public String filename = "C:\\Users\\macku\\Desktop\\Praca Magisterska\\Praca-Magisterska\\pila.txt";
// 	public String filename = "C:\\Users\\Maciej\\Desktop\\Praca Magisterska\\pila.txt";
	
// 	void writeLog(String s){
// 		try{
// 			FileWriter writer = new FileWriter(filename, true);
// 			// for (int[] row : invertedList) {
// 			// 	for (int value : row) {
// 			// 		writer.write(String.valueOf(value));
// 			// 		writer.write(" ");
// 			// 	}
// 			// 	writer.write("\n");
// 			// }
// 			writer.write(s);
// 			writer.write("\n...\n");
// 			writer.close();
// 		} catch(IOException e){
	
// 		}
// 	}

// 	void writeLogArray(double[] s){
// 		try{
// 			FileWriter writer = new FileWriter(filename, true);
	
// 			for (double value : s) {
// 				writer.write(String.valueOf(value));
// 				writer.write(" ");
// 			}
// 			writer.write("\n...\n");
// 			writer.close();
// 		} catch(IOException e){
	
// 		}
// 	}

// 	void writeLogArray2D(double[][] s){
// 		try{
// 			FileWriter writer = new FileWriter(filename, true);
// 			for (double[] row : s) {
// 				for (double value : row) {
// 					writer.write(String.valueOf(value));
// 					writer.write(" ");
// 				}
// 				writer.write("\n");
// 			}
// 			writer.write("\n...\n");
// 			writer.close();
// 		} catch(IOException e){
	
// 		}
// 	}

// 	static final long serialVersionUID = -1;

// 	// precision 
// 	final static double EPSILON = 1e-8;
	
// 	// regularization constant
// 	public double R = 10.0;
	
// 	// regularization constant per probability
// 	public double Rp = 1e-2;

// 	// inverted lists
// 	protected int[][] invertedList = null; 

// 	protected Instances instances = null;
	
// 	protected int D;
	
// 	protected int indexAttribute;
	
// 	protected int K;
	
// 	protected int N;
	
// 	protected boolean useLineSearch = false;
	
// 	protected double nu = 0.1;
	
// 	protected double[][] f = null;
	
// 	private double[][] probability = null;
	
// 	private double gradient = 0;
	
// 	private double hessian = 0;
	
// 	private int maxK = 0;
	
// 	public boolean preChosenK = false;
	
// 	private double[] gradients = null;
	
// 	private double[] hessians = null;
	
// 	private boolean useGradient = true;
	
// 	private double lineSearchMax = 4.0;
	
// 	private int lineSearchIterations = 10;
	
// 	private double lineSearchPrecision = 1e-4;
	
// 	public RuleBuilder(double nu, boolean useLineSearch, boolean useGradient, boolean preChosenK, double R, double Rp) {
// 		this.nu = nu;
// 		this.useLineSearch = useLineSearch;
// 		this.useGradient = useGradient;
// 		this.preChosenK = preChosenK;
// 		this.R = R;
// 		this.Rp = Rp;
// 		// writeLog("nu" + String.valueOf(nu));
// 		// writeLog("useLineSearch" + String.valueOf(useLineSearch));
// 		// writeLog("useGradient" + String.valueOf(useGradient));
// 		// writeLog("preChosenK" + String.valueOf(preChosenK));
// 		// writeLog("R" + String.valueOf(R));
// 		// writeLog("Rp" + String.valueOf(Rp));
// 	}

// 	protected final class Cut {

// 		public double[] decision = null;
// 		public int position = -1;
// 		public int direction = 0;
// 		public double value = 0;
// 		public double empiricalRisk = 0;
// 		public boolean exists = false;
	
// 		Cut(int K){
// 			super();
// 			decision = new double[K];
// 		}
		
// 		Cut(Cut cut){
// 			decision = new double[cut.decision.length];
// 			copy(cut);
// 		}
		
// 		void copy(Cut cut){
// 			for (int i = 0; i < decision.length; i++)
// 				decision[i] = cut.decision[i];
// 			position = cut.position;
// 			direction = cut.direction;
// 			value = cut.value;
// 			exists = cut.exists;
// 			empiricalRisk = cut.empiricalRisk;
// 		}
		
// 		public void saveCut(int direction, double currentValue, double value, double tempEmpiricalRisk) {
// 			this.direction = direction;
// 			this.value = (currentValue + value)/2;
// 			this.empiricalRisk = tempEmpiricalRisk;
// 			this.exists = true;
// 		}
// 	}

// 	public void initialize(Instances instances) {
		
// 		this.instances = instances;
// 		N = instances.numInstances();
// 		D = instances.numAttributes() - 2;
// 		K = instances.numClasses();
// 		indexAttribute = D + 1;
				
// 		invertedList = new int[D][N];
		
// 		for(int j = 0; j < D; j++) {
// 			instances.sort(j);
// 			double[] temporaryDoubleArray = instances.attributeToDoubleArray(indexAttribute);
// 			for(int i = 0; i < N; i++) {
// 				invertedList[j][i] = (int)temporaryDoubleArray[i]; 
// 			}
// 		}
		
// 		instances.sort(indexAttribute);

		
// 		probability = new double[N][K];
// 		gradients = new double[K];
// 		hessians = new double[K];
// 	}
	
// 	private void initializeForCut() {
// 		gradient = 0;
// 		hessian = R;
// 		Arrays.fill(gradients, 0);
// 		Arrays.fill(hessians, R);
// 	}
				
// 	public void initializeForRule(double[][] f, short[] coveredInstances) {
// 		this.f = f;
// 		if (preChosenK) {
// 			Arrays.fill(gradients, 0.0);
// 			Arrays.fill(hessians, R);
// 		}
		
// 		for (int i = 0; i < N; i++)
// 			if(coveredInstances[i] >= 0) {
// 				double norm = 0;
// 				for (int k = 0; k < K; k++) {
// 					probability[i][k] = Math.exp(f[i][k]);
// 					norm += probability[i][k];
// 				}
// 				for (int k = 0; k < K; k++) {
// 					probability[i][k] /= norm;
// 					if (preChosenK) {
// 						gradients[k] -= instances.instance(i).weight() * probability[i][k];
// 						hessians[k] += instances.instance(i).weight() * (Rp + probability[i][k] * (1 - probability[i][k]));
// 					}
// 				}
// 				// writeLog("temp gradients");
// 				// writeLogArray(gradients);
// 				if (preChosenK)
// 					// writeLog("y:");
// 					// writeLog(String.valueOf((int)instances.instance(i).classValue()));
// 					gradients[(int)instances.instance(i).classValue()] += instances.instance(i).weight();
// 			}
		
// 		if (preChosenK) {
// 			maxK = 0;
// 			if (useGradient) {
// 				// writeLog("Gradients:");
// 				// writeLogArray(gradients);
// 				for (int k = 1; k < K; k++)
// 					if (gradients[k] > gradients[maxK]){
// 						writeLog("podstawienie k za max_k. w if K:");
// 						writeLog(String.valueOf(k));
// 						maxK = k;
// 					}
// 			}
// 			else {
// 				for (int k = 1; k < K; k++)
// 					if (gradients[k] / Math.sqrt(hessians[k]) > gradients[maxK] / Math.sqrt(hessians[maxK])){
// 						writeLog("podstawienie k za max_k. w else K:");
// 						writeLog(String.valueOf(k));
// 						maxK = k;
// 					}
// 			}
// 		}
// 	}
		
// 	public double computeCurrentEmpiricalRisk(int position, int weight) {
		
// 		if (preChosenK) {
// 			if ((int)instances.instance(position).classValue() == maxK) 
// 				gradient += instances.instance(position).weight() * weight;
// 			gradient -= instances.instance(position).weight() * weight * probability[position][maxK];
// 			// writeLog("gradient: " + String.valueOf(gradient));
// 			if (useGradient)
// 				return -gradient;
// 			else {
// 				hessian += instances.instance(position).weight() * weight * (Rp + probability[position][maxK] * (1 - probability[position][maxK]));
// 				return - gradient * Math.abs(gradient) / hessian;
// 			}
// 		}
// 		else {
// 			int y = (int)instances.instance(position).classValue();
// 			for (int k = 0; k < K; k++) {
// 				if (y == k)
// 					gradients[k] += instances.instance(position).weight() * weight;
// 				gradients[k] -= instances.instance(position).weight() * weight * probability[position][k];
// 				if (!useGradient) {
// 					hessians[k] += instances.instance(position).weight() * weight * (Rp + probability[position][k] * (1 - probability[position][k]));
// 				}
// 			}
			
// 			if (useGradient) {
// 				double highest = gradients[0];
// 				for (int k = 1; k < K; k++)
// 					highest = Math.max(highest, gradients[k]);
// 				return - highest;
// 			}
// 			else {
// 				double highest = gradients[0] * Math.abs(gradients[0]) / hessians[0];
// 				for (int k = 1; k < K; k++)
// 					highest = Math.max(highest, gradients[k] * Math.abs(gradients[k]) / hessians[k]);
// 				return - highest;				
// 			}
// 		}
// 	}
	
// 	public double[] computeDecision(short[] coveredInstances) {

// 		if (preChosenK) {
// 			hessian = R;
// 			gradient = 0;
// 			writeLog("halko prechosenK wybrane");

// 			writeLog("probability");
// 			writeLogArray2D(probability);

// 			double[] transformed = new double[N];
// 			for (int j=0;j<N;j++) {
// 				transformed[j] = (double)coveredInstances[j];
// 			}
// 			// writeLog("covered Instances:");
// 			// writeLogArray(transformed);
		
// 			for (int i = 0; i < coveredInstances.length; i++){
// 				if (coveredInstances[i] >= 0) {
// 					if ((int)this.instances.instance(i).classValue() == maxK)
// 						gradient += instances.instance(i).weight();
// 					gradient -= instances.instance(i).weight() * probability[i][maxK];
// 					hessian += instances.instance(i).weight() * (Rp + probability[i][maxK] * (1 - probability[i][maxK]));
// 					if (instances.instance(i).weight() != 1){
// 						writeLog("weights");
// 						writeLog(String.valueOf(instances.instance(i).weight()));	
// 					}
// 				}
// 			}

// 			writeLog(String.valueOf(hessian));
// 			writeLog(String.valueOf(gradient));
		
// 			if (gradient <= 0)
// 				return null;

// 			writeLog("Max_K:");
// 			writeLog(String.valueOf(maxK));
			
// 			double alphaNR = gradient / hessian;
// 			writeLog("alphanr:");
// 			writeLog(String.valueOf(alphaNR));
// 			double[] decision = new double[K];
// 			Arrays.fill(decision, - alphaNR / K);
// 			decision[maxK] = alphaNR * (K - 1) / K;
// 			writeLog("decision z compute decision na koncu:");
// 			writeLogArray(decision);
// 			return decision;
// 		}
// 		else {
// 			writeLog("halko prechosenK NIE wybrane");
// 				Arrays.fill(hessians, R);
// 			Arrays.fill(gradients, 0);
// 			int chosenK = 0;
// 			double[] origGradients = new double[K];
		
// 			for (int i = 0; i < coveredInstances.length; i++)
// 				if (coveredInstances[i] >= 0) {
// 					for (int k = 0; k < K; k++) {
// 						if ((int)this.instances.instance(i).classValue() == k) {
// 							gradients[k] += instances.instance(i).weight();
// 							origGradients[k] += instances.instance(i).weight() * coveredInstances[i];
// 						}
// 						gradients[k] -= instances.instance(i).weight() * probability[i][k];
// 						hessians[k] += instances.instance(i).weight() * (Rp + probability[i][k] * (1 - probability[i][k]));
// 						origGradients[k] -= instances.instance(i).weight() * coveredInstances[i] * probability[i][k];
// 					}
// 				}
// 			// writeLogArray(gradients);
// 			// writeLogArray(hessians);
// 			// writeLogArray(origGradients);
		
// 			for (int k = 1; k < K; k++)
// 				if (origGradients[k] > origGradients[chosenK])
// 					chosenK = k;
			
// 			if (gradients[chosenK] <= 0)
// 				return null;
			
// 			double alphaNR = gradients[chosenK] / hessians[chosenK];
// 			double[] decision = new double[K];
// 			Arrays.fill(decision, - alphaNR / K);
// 			decision[chosenK] = alphaNR * (K - 1) / K;
// 			return decision;
// 		}
// 	}
	
// 	public Cut findBestCut(int attribute, short[] coveredInstances) {	
// 		Cut bestCut = new Cut(K);
// 		bestCut.position = -1; 
// 		bestCut.exists = false;
// 		bestCut.empiricalRisk = 0;
			
// 		double tempEmpiricalRisk = 0;

// 		for(int cutDirection = -1; cutDirection <= 1; cutDirection += 2) {			
// 			initializeForCut();
		
// 			int currentPosition = 0;
// 			int i;
// 			i = (cutDirection == Rule.GREATER_EQUAL)? N - 1 : 0;
// 			while((cutDirection == Rule.GREATER_EQUAL && i >= 0) || (cutDirection != Rule.GREATER_EQUAL && i < N)) {
// 				currentPosition = invertedList[attribute][i];
// 				if(coveredInstances[currentPosition] > 0 && instances.instance(currentPosition).isMissing(attribute) == false)
// 					break;
// 				if(cutDirection == Rule.GREATER_EQUAL) 
// 					i--; 
// 				else i++;
// 			}
		
// 			double currentValue = instances.instance(currentPosition).value(attribute);
// 			// writeLog("...");
// 			// writeLog("...");
// 			// writeLog("current_value");
// 			// writeLog(String.valueOf(currentValue));
// 			// writeLog("current position");
// 			// writeLog(String.valueOf(currentPosition));
// 			// writeLog("i:");
// 			// writeLog(String.valueOf(i));
// 			while((cutDirection == Rule.GREATER_EQUAL && i >= 0) || (cutDirection != Rule.GREATER_EQUAL && i < instances.numInstances())) {
// 				int nextPosition = invertedList[attribute][i];

// 				// writeLog("next position: " + String.valueOf(nextPosition));
				
// 				if(coveredInstances[nextPosition] > 0) {
// 					if(instances.instance(nextPosition).isMissing(attribute) == false) {
// 						double value = instances.instance(nextPosition).value(attribute);
// 						// writeLog("value: " + String.valueOf(value));
// 						if (currentValue != value) 
// 							if (tempEmpiricalRisk < bestCut.empiricalRisk + RuleBuilder.EPSILON)
// 								bestCut.saveCut(cutDirection, currentValue, value, tempEmpiricalRisk);
// 						tempEmpiricalRisk = computeCurrentEmpiricalRisk(nextPosition, coveredInstances[nextPosition]);
// 						// writeLog("temp empirical risk" + String.valueOf(tempEmpiricalRisk));
// 						currentValue = instances.instance(nextPosition).value(attribute);
// 					}
// 				}
// 				if(cutDirection == Rule.GREATER_EQUAL) i--; else i++;
// 			}
// 			// writeLog("End of calculating");
// 		}
// 		// writeLog("decisiom:");
// 		// writeLogArray(bestCut.decision);
// 		// writeLog("position:");
// 		// writeLog(String.valueOf(bestCut.position));
// 		// writeLog("direction:");
// 		// writeLog(String.valueOf(bestCut.direction));
// 		// writeLog("value:");
// 		// writeLog(String.valueOf(bestCut.value));
// 		// writeLog("empirical_risk:");
// 		// writeLog(String.valueOf(bestCut.empiricalRisk));
// 		// writeLog("exists:");
// 		// writeLog(String.valueOf(bestCut.exists));

// 		return bestCut;
// 	}
	
// 	public short[] markCoveredInstances(int bestAttribute, short[] coveredInstances, Cut cut) { 
// 		for(int i = 0; i < N; i++) 
// 			if (coveredInstances[i] != -1) {
// 				if(instances.instance(i).isMissing(bestAttribute) == true)
// 					coveredInstances[i] = -1;
// 				else {
// 					double value = instances.instance(i).value(bestAttribute);
// 					if (((value < cut.value) && (cut.direction == 1)) ||
// 						((value > cut.value) && (cut.direction == -1)))
// 						coveredInstances[i] = -1;
// 				}
// 			}
// 		return coveredInstances;
// 	}
	
// 	private double computeRuleGradient(double[][] f, short[] coveredInstances, double point) {
// 		int size = 0;
// 		double gradient = 0;
// 		for (int i = 0; i < N; i++)
// 			if(coveredInstances[i] >= 0) {
// 				size++;
// 				probability[i][maxK] = Math.exp(f[i][maxK] + point);
// 				double norm = probability[i][maxK];
// 				for (int k = 0; k < K; k++)
// 					if (k != maxK) {
// 						probability[i][k] = Math.exp(f[i][k]);
// 						norm += probability[i][k];
// 					}
// 				if ((int)instances.instance(i).classValue() == maxK)
// 					gradient += instances.instance(i).weight();
// 				gradient -= instances.instance(i).weight() * probability[i][maxK] / norm;
// 			}
// 		return gradient / size;
// 	}
	
// 	public double[] getLineSearchDecision(double[][] f, short[] coveredInstances) {
// 		double gradient = computeRuleGradient(f, coveredInstances, lineSearchMax);
// 		if (gradient >= 0) {
// 			double[] decision = new double[K];
// 			Arrays.fill(decision, - lineSearchMax / K);
// 			decision[maxK] = lineSearchMax * (K - 1) / K;
// 			return decision;			
// 		}
// 		else
// 			return getLineSearchDecision(f, coveredInstances, 0, lineSearchMax, 1);
// 	}
	
// 	private double[] getLineSearchDecision(double[][] f, short[] coveredInstances, 
// 			double left, double right, int depth) {
// 		double middle = (left + right) / 2.0;
// 		double gradient = computeRuleGradient(f, coveredInstances, middle);		
// 		if (Math.abs(gradient) < lineSearchPrecision || depth == lineSearchIterations) {
// 			double[] decision = new double[K];
// 			Arrays.fill(decision, - middle / K);
// 			decision[maxK] = middle * (K - 1) / K;
// 			return decision;			
// 		}
// 		else {
// 			if (gradient > 0) 
// 				return getLineSearchDecision(f, coveredInstances, 
// 						middle, right, depth + 1);
// 			else
// 				return getLineSearchDecision(f, coveredInstances, 
// 						left, middle, depth + 1);
// 		}	
// 	}
	
// 	public Rule createRule(double[][] f, short[] coveredInstances) {
		
// 		initializeForRule(f, coveredInstances);

// 		Rule rule = new Rule(instances.classAttribute());

// 		Cut bestCut = new Cut(K);
// 		bestCut.empiricalRisk = 0;
// 		boolean creating = true;

// 		while(creating){
// 			int bestAttribute = -1;
// 			Cut cut;
// 			for (int j = 0; j < D; j++) {
// 				cut = findBestCut(j, coveredInstances);
// 				if (cut.empiricalRisk < bestCut.empiricalRisk - RuleBuilder.EPSILON) {
// 					bestCut.copy(cut);
// 					bestAttribute = j;
// 				}
// 			}			
// 			if(bestAttribute == -1 || bestCut.exists == false) {
// 				creating = false;
// 			}
// 			else {
// 				rule.addSelector(bestAttribute, bestCut.value, bestCut.direction, instances.attribute(bestAttribute));
// 				coveredInstances = markCoveredInstances(bestAttribute, coveredInstances, bestCut);
// 			}
// 		}
				
// 		if (bestCut.exists == true) {
// 			double[] decision = null;
// 			if (useLineSearch)
// 				decision = getLineSearchDecision(f, coveredInstances);
// 			else
// 				decision = computeDecision(coveredInstances);
// 			if (decision == null)
// 				return null;
// 			for (int i = 0; i < decision.length; i++)
// 				decision[i] *= nu;
// 			rule.setDecision(decision);
// 			return rule;
// 		}
// 		else 
// 			return null;
// 	}
	
// 	public double[] createDefaultRule(double[][] f, short[] coveredInstances) {
// 		initializeForRule(f, coveredInstances);
// 		double[] decision = computeDecision(coveredInstances);
// 		for (int i = 0; i < decision.length; i++)
// 			decision[i] *= nu;
// 		writeLog("decision:");
// 		writeLogArray(decision);
// 		return decision;
// 	}
	
// 	public double[] createDefaultRule() {
// 		double[] priors = new double[K];
// 		for (int i = 0; i < N; i++)
// 			priors[(int)instances.instance(i).classValue()]++;
// 		int emptyClasses = 0;
// 		for (int k = 0; k < K; k++) {
// 			priors[k] /= (double)N;
// 			if (priors[k] == 0)
// 				emptyClasses++;
// 		}
// 		double logPriors = 0;
// 		for (int k = 0; k < K; k++)
// 			if (priors[k] != 0)
// 				logPriors += Math.log(priors[k]);
// 		logPriors /= (K - emptyClasses);
		
// 		double[] decision = new double[K];
// 		Arrays.fill(decision, -logPriors);
// 		for (int k = 0; k < K; k++)
// 			if (priors[k] != 0)
// 				decision[k] += Math.log(priors[k]);
// 			else
// 				decision[k] = 0;
// 		return decision;
// 	}

// }
/*
 * Created on 2007-07-05
 * 
 */

 package put.idss.mlrules;

 import java.io.Serializable;
 import java.util.Arrays;
 import java.util.Comparator;
 
 import weka.core.FindWithCapabilities;
 import weka.core.Instances;
 
 public final class RuleBuilder implements Serializable {
     
     protected final static double EPSILON = 1e-8;
     
     public final static double MISSING_VALUE = 1e20;
         
     protected int maxCuts = -1;
     
     protected double[][] thresholds = null; 
     protected double[][] statistics = null; 
     protected double[] missingStatistics = null; 
     protected int[][] invertedList = null;
 
     protected int D;
     
     protected int K;
     
     protected int N;
     
     protected double[][] x;
 
     protected int[] y;
     
     public double[] w;
     
     public double[] gradients;
 
     protected double nu = 0.1;
     
     protected double[] f = null;
     
     public double gradientInitial = 0;
     
     protected Instances instances;
     
     private int bestCut_direction = 0;
     private double bestCut_value = 0;
     private double bestCut_empiricalRisk = 0;
     private int bestCut_attribute = 0;
     private boolean bestCut_exists = false;
     
     public static final double R = 0.5;
     
     private double alpha = 1.0;
     private double beta = 0.0;
     
     public RuleBuilder(double nu, int maxCuts, double alpha, double beta) {
         this.nu = nu;
         this.maxCuts = maxCuts;
         this.alpha = alpha;
         this.beta = beta;
     }
 
     public final void initialize(double[][] x, int[] y, double[] w, Instances instances) {
 
         this.x = x;
         this.y = y;
         N = x.length;
         D = x[0].length;
 
         this.w = w;
         
         this.instances = instances;
                 
         invertedList = new int[N][D];
         
         thresholds = new double[D][];
         statistics = new double[D][];
         missingStatistics = new double[D];
 
         double[] tempThresholds = new double[N+1];
         int[] y_stat = new int[N];
         double[][] temp_x = new double[N][2];
         
         
         for(int j = 0; j < D; j++) {
             for (int i = 0; i < N; i++) {
                 temp_x[i][0] = i;
                 temp_x[i][1] = x[i][j];
             }
             Arrays.sort(temp_x, new Comparator<double[]>() {
                 public int compare(double[] o1, double[] o2) {
                     return (o1[1] - o2[1] > 0 ? 1 : (o1[1] - o2[1] < 0 ? -1 : 0));
                 } 
             });
             
             double thr_previous = -1e20;
             int numThresholds = 0;
             for (int i = 0; i < N; i++) 
                 if (temp_x[i][1] < MISSING_VALUE) {
                     int curr_index = (int)temp_x[i][0];				
                     if (Math.abs(temp_x[i][1] - thr_previous) > EPSILON) {
                         numThresholds++;
                         tempThresholds[numThresholds-1] = (temp_x[i][1] + thr_previous) / 2;
                         y_stat[numThresholds-1] = y[curr_index];
                     }
                     else
                         if (y_stat[numThresholds-1] != y[curr_index])
                             y_stat[numThresholds-1] = 0;
                     thr_previous = temp_x[i][1];
                 }
             
             // remove thresholds of the same size
             
             int offset = 0;
             int t = 0;
             while (t < numThresholds) {
                 int z = t + 1;
                 while (z < numThresholds && y_stat[z] == y_stat[t] && y_stat[t] != 0)
                     z++;
                 tempThresholds[offset] = tempThresholds[t];
                 t = z;
                 offset++;
             }
             numThresholds = offset;
             
             tempThresholds[numThresholds] = Double.MAX_VALUE;
             
             thresholds[j] = new double[numThresholds];
             for (int i = 0; i < numThresholds; i++)
                 thresholds[j][i] = tempThresholds[i];
             statistics[j] = new double[numThresholds];
             
             int thr_index = 1;
             for (int i = 0; i < N; i++) {
                 int curr_index = (int)temp_x[i][0];
                 if (temp_x[i][1] > MISSING_VALUE)
                     invertedList[curr_index][j] = -1;
                 else {
                     if (temp_x[i][1] > tempThresholds[thr_index])
                         thr_index++;
                     invertedList[curr_index][j] = thr_index - 1;
                 }
             }
             //System.out.println("num distinct " + numThresholds + " (" + (100.0 * numThresholds / N) + ")");
 
             /*
             int numDistinct = 0;
             double thr_previous = -1e10;
             int y_prev = 0;
             for(int i = 0; i < N; i++) {
                 int curr_index = (int)temp_x[i][0];
                 if (Math.abs(temp_x[i][1] - thr_previous) > 1e-12 && temp_x[i][1] < MISSING_VALUE) {
                     numDistinct++;
                 }
                 if (temp_x[i][1] < MISSING_VALUE)
                     invertedList[curr_index][j] = numDistinct - 1;
                 else
                     invertedList[curr_index][j] = -1;
                 y_prev = y[curr_index];
                 thr_previous = temp_x[i][1];
             }			
             System.out.println("num distinct " + numDistinct + " (" + (100.0 * numDistinct / N) + ")");
             thresholds[j] = new double[numDistinct];
             statistics[j] = new double[numDistinct];
             numDistinct = 0;
             thr_previous = -1e10;
             y_prev = 0;
             for(int i = 0; i < N; i++) {
                 int curr_index = (int)temp_x[i][0];
                 if (Math.abs(temp_x[i][1] - thr_previous) > 1e-12  && temp_x[i][1] < MISSING_VALUE) {
                     numDistinct++;
                     thresholds[j][numDistinct-1] = (temp_x[i][1] + thr_previous) / 2;
                 }
                 y_prev = y[curr_index];
                 thr_previous = temp_x[i][1];
             }
             */
         }
         
         gradients = new double[N];
     }
     
     public final void initializeStatistics(short[] coveredInstances) {
         gradientInitial = 0;
         Arrays.fill(missingStatistics, 0);
         for (int j = 0; j < D; j++)
             Arrays.fill(statistics[j], 0);
         for (int i = 0; i < N; i++)
             if (coveredInstances[i] == 1) {
                 for (int j = 0; j < D; j++)
                     if (invertedList[i][j] != -1)
                         statistics[j][invertedList[i][j]] += gradients[i];
                     else
                         missingStatistics[j] += gradients[i];
                 gradientInitial += gradients[i];
             }
     }
     
     public final void updateStatistics(short[] coveredInstances) {
         for (int i = 0; i < N; i++)
             if (coveredInstances[i] == -2) {
                 for (int j = 0; j < D; j++)
                     if (invertedList[i][j] != -1)
                         statistics[j][invertedList[i][j]] -= gradients[i];
                     else
                         missingStatistics[j] -= gradients[i];
                 gradientInitial -= gradients[i];
                 coveredInstances[i] = -1;
             }
     }
     
     protected void initializeForRule(double[] f, short[] coveredInstances) {
         this.f = f;
         
         for (int i = 0; i < N; i++) {
                 if (y[i] > 0)
                     gradients[i] = -w[i] / (1 + Math.exp(f[i]));
                 else
                     gradients[i] = alpha * w[i] / (1 + Math.exp(-alpha * f[i]-beta));
                 
             }
     }
         
     protected double computeDecision(short[] coveredInstances) {
 
         double W_plus = 0;
         double W_minus = 0;
         
         for (int i = 0; i < N; i++) 
             if (coveredInstances[i] >= 0) {
                 if (y[i] > 0)
                     W_plus -= gradients[i];
                 else
                     W_minus += gradients[i];
             }
         
         return 1.0 / (alpha + 1.0) * Math.log((W_plus + R) / (W_minus + alpha * R));
     }
     
     protected final void findBestCut(int attribute) {	
 
         double gradientLeft = statistics[attribute][0];
         double gradientRight = gradientInitial - missingStatistics[attribute] - gradientLeft;
 
         for (int i = 1; i < thresholds[attribute].length; i++) {
             if (-Math.abs(gradientLeft) < bestCut_empiricalRisk - EPSILON) {
                 bestCut_exists = true;
                 bestCut_attribute = attribute;
                 bestCut_direction = Rule.LESS_EQUAL;
                 bestCut_empiricalRisk = -Math.abs(gradientLeft);
                 bestCut_value = thresholds[attribute][i];
             }
             if (-Math.abs(gradientRight) < bestCut_empiricalRisk - EPSILON) {
                 bestCut_exists = true;
                 bestCut_attribute = attribute;
                 bestCut_direction = Rule.GREATER_EQUAL;
                 bestCut_empiricalRisk = -Math.abs(gradientRight);
                 bestCut_value = thresholds[attribute][i];
             }
             gradientLeft += statistics[attribute][i];
             gradientRight -= statistics[attribute][i];
         }
     }
     
     protected final void markCoveredInstances(int bestAttribute, short[] coveredInstances) { 
         for(int i = 0; i < N; i++) 
             if (coveredInstances[i] >= 0) {
                 if(x[i][bestAttribute] > MISSING_VALUE)
                     coveredInstances[i] = (short) -(coveredInstances[i] + 1);
                 else {
                     double value = x[i][bestAttribute];
                     if (((value < bestCut_value) && (bestCut_direction == 1)) ||
                         ((value > bestCut_value) && (bestCut_direction == -1)))
                         coveredInstances[i] = (short) -(coveredInstances[i] + 1);
                 }
             }
         updateStatistics(coveredInstances);
     }
     
     public final Rule createRule(double[] f, short[] coveredInstances) {
         
         initializeForRule(f, coveredInstances);
         initializeStatistics(coveredInstances);
 
         Rule rule = new Rule(instances.classAttribute());
 
         bestCut_empiricalRisk = 0;
         boolean creating = true;
         int numCutsUsed = 0;
         bestCut_exists = false;
 
         while(creating) {
             for (int j = 0; j < D; j++)
                 findBestCut(j);
             if (!bestCut_exists) {
                 creating = false;
             }
             else {
                 rule.addSelector(bestCut_attribute, bestCut_value, bestCut_direction, instances.attribute(bestCut_attribute));
                 markCoveredInstances(bestCut_attribute, coveredInstances);
             }
             numCutsUsed ++;
             if (numCutsUsed == maxCuts)
                 creating = false;
         }
                 
         if (bestCut_exists == true) {
             double decision = nu * computeDecision(coveredInstances);
             rule.setDecision(decision);
             return rule;
         }
         else {
             return null;
         }
     }
     
 
     public double createDefaultRule() {
         double totalWeightNegative = 0;
         double totalWeightPositive = 0;
         for (int i = 0; i < N; i++) {
             if (y[i] == 1)
                 totalWeightPositive += w[i];
             else
                 totalWeightNegative += w[i];
         }
         double initial = 0;
         for (int i = 0; i < 20; i++) {
             double q0 = 1.0 / (1.0 + Math.exp(-alpha * initial-beta));
             double q1 = 1.0 / (1.0 + Math.exp(-initial));
             initial -= (alpha * totalWeightNegative *q0 - totalWeightPositive * (1-q1)) 
                     / (alpha*alpha * totalWeightNegative * q0 * (1-q0) + totalWeightPositive * q1 * (1-q1));
         }
         return initial;
     }
     
     public double computeProbability(double value) {
         double expf = Math.exp(value);
         double expfalpha = Math.exp(alpha * value+beta);
         return alpha * expfalpha / (alpha * expfalpha + (1+expfalpha) / (1 + expf));
     }
     
     public double getLoss(double y_, double value) {
         if (y_ > 0)
             return Math.log(1+ Math.exp(- value));
         else
             return Math.log(1+ Math.exp(alpha * value + beta));
     }
 
 }