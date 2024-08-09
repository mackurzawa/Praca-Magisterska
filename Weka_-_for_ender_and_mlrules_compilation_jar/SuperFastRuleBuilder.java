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
 
 public final class SuperFastRuleBuilder implements Serializable {
     
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
     
     public SuperFastRuleBuilder(double nu, int maxCuts, double alpha, double beta) {
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