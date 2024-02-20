/*
 * Created on 2007-07-05
 * 
 */

package weka.classifiers.rules;

import java.util.Arrays;
import java.util.Enumeration;
import java.util.Random;
import java.util.Vector;

import put.idss.mlrules.Rule;
import put.idss.mlrules.RuleBuilder;

import weka.classifiers.*;
import weka.core.*;
import weka.core.Capabilities.Capability;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;

/**
<!-- globalinfo-start -->
* Maximum Likelihood Rule Ensembles (MLRules) - class for building a rule ensemble 
* for classification via estimating the conditional class probabilities.<br/> 
* Rules are combined in additive way.<br/>
* <br/>
* For more information, see:<br/>
* <br/>
* Krzysztof Dembczynski, Wojciech Kotlowski, Roman Slowinski:<br/>
* <i>Maximum Likelihood Rule Ensembles</i><br/>
* Proceedings of the 25th International Conference on Machine Learning (ICML 2008).<br/> 
* Helsinki, Finland.
* <p/>
<!-- globalinfo-end -->
*
<!-- technical-bibtex-start -->
* BibTeX:
* <pre>
* &#64;@inproceedings{Dembczynski_etal2008,
  author    = {Krzysztof Dembczy\'nski and
               Wojciech Kot{\l}owski and
               Roman S{\l}owi\'nski},
  title     = {Maximum Likelihood Rule Ensembles},
  booktitle = {Proceedings of the 25th International 
               Conference on Machine Learning (ICML 2008)},
  address   = {Helsinki, Finland},
  editor    = {Andrew McCallum and Sam Roweis},
  publisher = {Omnipress},
  year      = {2008},
  pages     = {224--231}
}
* </pre>
* <p/>
<!-- technical-bibtex-end -->
*
<!-- options-start -->
* Valid options are: <p/>
* 
* <pre> -M &lt;number of rules&gt;
*  Set the number of rules, i.e. the ensemble size.
*  (default 100)</pre>
* 
* <pre> -S &lt;shrinkage&gt;
*  Set the amount of shrinkage.
*  (default 0.5)</pre>
* 
* <pre> -R 
*  No resampling. (default resampling is on)</pre>
* 
* <pre> -P 
*  Set the size of the subsample (fraction of the training set). (default 0.5)</pre>
*
* <pre> -Q &lt;technique&gt;
*  Set the minimization technique:
*    0 = gradient descent,
*    1 = Newton-Raphson.
*  (default 0) </pre>
* 
<!-- options-end -->
*
* @author Wojciech Kotlowski (wkotlowski@cs.put.poznan.pl)
* @author Krzysztof Dembczynski (kdembczynski@cs.put.poznan.pl)
*/


public class MLRules extends Classifier implements OptionHandler, TechnicalInformationHandler{
	
	static final long serialVersionUID = -1;
	
	public static int MINIMIZER_GRADIENT = 0;
	public static int MINIMIZER_NEWTON = 1;
	
	public static final Tag [] TAGS_MINIMIZER = {
		new Tag(MINIMIZER_GRADIENT, "Gradient descent"),
		new Tag(MINIMIZER_NEWTON, "Newton-Raphson step")
	};
	
	private boolean modelBuilt = false;
	

	/**
	 * covered instances 
	 */
	private short[] coveredInstances = null;
	
	/**
	 * decision rules
	 */
	private Rule[] rules;
	
	/**
	 * filter for binarization of nominal attributes
	 */
	private NominalToBinary ntb;
	
	/**
	 * instances
	 */
	private Instances instances;
	
	/**
	 *  number of instances
	 */
	private int N = 0;

	/**
	 * number of attributes
	 */
	private int D = 0;
		
	/**
	 * number of classes
	 */
	private int K = 0;
	
	/**
	 * number of rules to be generated
	 */
	private int nRules = 100;
	
	/**
	 * value of default rule
	 */
	private double[] defaultRule = null;
	
	/**
	 * current function values
	 */
	private double[][] f;
	
	/**
	 * rule builder
	 */
	private RuleBuilder ruleBuilder = null;
	
	/**
	 * whether resampling is performed
	 */
	private boolean resample = true;
	
	/**
	 * percentage of resampled data
	 */
	private double percentage = 0.5;
	
	/**
	 * shrinkage
	 */
	private double nu = 0.5;
	
	/**
	 * Whether to use line search (set to false forever)
	 */
	private boolean useLineSearch = false;
		
	/**
	 * Minimization technique
	 */
	private int minimization = MINIMIZER_GRADIENT;
	
	/**
	 * Whether a given class should be chosen before rule generation
	 * (set to true forever)
	 */
	private boolean chooseClass = true;
	
	/**
	 * Parameter penalizing small rules
	 */
	private double R = 5.0;
	
	/**
	 * Parameter penalizing small rules
	 */
	private double Rp = 1e-5;
	
	/**
	 * random number generator
	 */
	private Random mainRandomGenerator = null;
	
	/**
	 * Returns an instance of a TechnicalInformation object, containing 
	 * detailed information about the technical background of this class,
	 * e.g., paper reference or book this class is based on.
	 * 	
	 * @return the technical information about this class
	 */
	public TechnicalInformation getTechnicalInformation() {
		
		TechnicalInformation result;
		result = new TechnicalInformation(Type.INPROCEEDINGS);
		
		result.setValue(Field.AUTHOR, "Krzysztof Dembczy�ski and Wojciech Kot�owski and Roman S�owi�ski");
		result.setValue(Field.TITLE, "Maximum likelihood rule ensembles");
		result.setValue(Field.BOOKTITLE, "Proceedings of the 25th International " + 
               "Conference on Machine Learning (ICML 2008)");
		result.setValue(Field.YEAR, "2008");
		result.setValue(Field.PAGES, "224--231");
		result.setValue(Field.ADDRESS, "Helsinki, Finland");
		result.setValue(Field.PUBLISHER, "Omnipress");
		return result;
	}
	
	/**
	 * Returns a string describing classifier
	 * @return a description suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String globalInfo() {

		return "Maximum Likelihood Rule Ensembles (MLRules) - class for building a rule ensemble "
		+ "for classification via estimating the conditional class probabilities.\n" 
		+ "Rules are combined in additive way.\n\n" 
		+ getTechnicalInformation().toString();	    
	}

	/**
	 * Returns default capabilities of the classifier.
	 * @return the capabilities of this classifier
	 */
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();
		
		result.enable(Capability.NOMINAL_ATTRIBUTES);
		result.enable(Capability.BINARY_ATTRIBUTES);
	    result.enable(Capability.MISSING_VALUES);
	    result.enable(Capability.NUMERIC_ATTRIBUTES);
	    result.enable(Capability.NOMINAL_CLASS);

		// instances
		result.setMinimumNumberInstances(1);

		return result;
	}	
	
	/**
	 * Returns a description of the classifier.
	 * @return a description of the classifier as a string.
	 */
	public String toString() {

		if (!modelBuilt) {
			return "Maximum Likelihood Rule Ensembles (MLRules): No model built yet.";
		}
		else {
			StringBuffer buffer = new StringBuffer("Maximum Likelihood Rule Ensembles (MLRules)...\n\n" +
					                               nRules + " rules generated.\n" +
					                               "Default rule:\n" + printDefaultRule() + "\n\n" +
					                               "List of rules:\n\n");
			for (int i = 0; i < nRules; i++)
				buffer.append(getRules()[i].toString() + "\n");
			return buffer.toString();
		}
	}
	
	/**
	 * Prints default rule in a readable way
	 * @return a description of the default rule
	 */
	private String printDefaultRule() {
		double[] defaultRule = getDefaultRule();
		StringBuffer ruleString = new StringBuffer();
		for (int i = 0; i < defaultRule.length; i++)
			ruleString.append("vote for class " + instances.classAttribute().value(i) + " with weight " + defaultRule[i] + "\n");
		return ruleString.toString();
	}


	
	public Instances getInstances() {
		return instances;
	}

	public Rule[] getRules() {
		return this.rules;
	}
	
	public double[] getDefaultRule() {
		return this.defaultRule;
	}
		
	public void setnRules(int nRules) {
		this.nRules = nRules;
	}
	
	public int getnRules() {
		return this.nRules;
	}
	
	public String nRulesTipText() {
		return "The total number of rules.";
	}

	
	public double[] getF(int position) {
		return f[position];
	}
	
	public int getD() {
		return D;
	}

	public int getK() {
		return K;
	}
		
	public MLRules() {
	}
	
	public short[] resample (double percentage) {
		
		short [] subSample = new short[N];
		
		int subsampleSize = (int) (N * percentage);

		Random random = new Random(mainRandomGenerator.nextInt());
		
		int[] indices = new int[N];
		for (int i = 0; i < N; i++)
			indices[i] = i;
		
	    for (int i = N - 1; i > 0; i--) {
	        int temp = indices[i];
	        int index = random.nextInt(i+1);
	        indices[i] = indices[index];
	        indices[index] = temp;
	    }

		for (int i = 0; i < subsampleSize; i++)
			subSample[indices[i]] = 1;				
		
		return subSample;
	}

	/**
	 * Generates the classifier.
	 *
	 * @param instances set of instances serving as training data 
	 * @throws Exception if the classifier has not been generated successfully
	 */
	public void buildClassifier(Instances instances) throws Exception {
		
		// can classifier handle the data?
	    getCapabilities().testWithFail(instances);
	    
		initialize(instances);
		rules = new Rule[nRules];
		Arrays.fill(coveredInstances, (short) 1);
		if (useLineSearch)
			defaultRule = ruleBuilder.createDefaultRule();
		else
			defaultRule = ruleBuilder.createDefaultRule(f, coveredInstances);
		updateFunction(defaultRule);
		for (int m = 0; m < nRules; m++) {
			if (resample == true)
				coveredInstances = resample(getPercentage());
			else
				Arrays.fill(coveredInstances, (short) 1);
			rules[m] = ruleBuilder.createRule(f, coveredInstances);

			if (rules[m] != null) {
				updateFunction(rules[m].getDecision());
			}
			else {
				m--;
			}
		}
	    modelBuilt = true;
	}

	private void initialize(Instances instances) throws Exception{
		
		this.instances = new Instances(instances);
						
		ntb = new NominalToBinary();
		ntb.setBinaryAttributesNominal(true);
		try {
			ntb.setInputFormat(this.instances);
			this.instances = Filter.useFilter(this.instances, ntb);
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		D = this.instances.numAttributes() - 1;
		N = this.instances.numInstances();
		K = this.instances.numClasses();
		f = new double[N][K];

		this.instances.insertAttributeAt(new Attribute("InstanceIndex"), D + 1);
		int indexAttribute = D + 1;
		for (int i = 0; i < N; i++)
			this.instances.instance(i).setValue(indexAttribute, i);
		
		coveredInstances = new short[N];

		ruleBuilder = new RuleBuilder(nu, useLineSearch, (minimization == 0), chooseClass, R, Rp);
		ruleBuilder.initialize(this.instances);
		
		mainRandomGenerator = new Random();
	}
	
	public void updateFunctionWhenRemoval(Rule rule) {
		for (int i = 0; i < N; i++)
			if (rule.classifyInstance(instances.instance(i)) != null)
				for (int k = 0; k < K; k++)
					f[i][k] -= rule.getDecision()[k];		
	}
	
	public void updateFunction(double[] decision)  {
		for (int i = 0; i < N; i++)
			if(coveredInstances[i] >= 0)
				for (int k = 0; k < K; k++)
					f[i][k] += decision[k];
	}

	public double[] evaluateF(Instance instance) {
		
		double [] evalF = new double[K];

		ntb.input(instance); 
		instance = ntb.output();
		for (int k = 0; k < K; k++)
			evalF[k] = defaultRule[k];

		for (int m = 0; m < nRules; m++) {
			double[] currentValues = rules[m].classifyInstance(instance);
			if (currentValues != null)
				for (int k = 0; k < K; k++)
					evalF[k] += currentValues[k]; 
		}
		return evalF;
	}
	
	/**
	 * Calculates the class membership probabilities for the given test instance.
	 *
	 * @param instance the instance to be classified
	 * @return predicted class probability distribution
	 * @throws Exception if class is numeric
	 */
	public double [] distributionForInstance(Instance instance) throws Exception {
		double[] evalF = evaluateF(instance);
		double[] distribution = new double[K];
			double total = 0;
			for (int k = 0; k < K; k++) {
				distribution[k] = Math.exp(evalF[k]);
				total += distribution[k];
			}
			for (int k = 0; k < K; k++)
				distribution[k] /= total;
			
			return distribution;
	}
	
	/**
	 * Returns an enumeration describing the available options.
	 *
	 * @return an enumeration of all the available options.
	 */
	public Enumeration<Option> listOptions() {
		Vector<Option> result = new Vector<Option>();

		result.addElement(new Option(
				"\tSet the number of rules, i.e. the ensemble size (default 100).",
				"M", 1, "-M <number of rules>"));

		result.addElement(new Option(
				"\tSet the amount of shrinkage (default 0.5).",
				"S", 1, "-S <shrinkage>"));
		
		result.addElement(new Option(
				"\tNo resampling (default resampling is on).",
				"R", 0, "-R"));
		
		result.addElement(new Option(
				"\tSet the size of the subsample as a fraction of the training set (default 0.5).",
				"P", 1, "-P"));

		result.addElement(new Option(
				"\tSet the minimization technique:\n" 
				+ "\t\t0 = gradient deccent,\n"
				+ "\t\t1 = Newton-Raphson.",
				"Q", 1, "-Q <technique>"));
		
		return result.elements();
	}
	
	/**
	 * Parses a given list of options. <p/>
	 *
	   <!-- options-start -->
	 * Valid options are: <p/>
	 * 
	 * <pre> -M &lt;number of rules&gt;
	 *  Set the number of rules, i.e. the ensemble size.
	 *  (default 100)</pre>
	 * 
	 * <pre> -S &lt;shrinkage&gt;
	 *  Set the amount of shrinkage.
	 *  (default 0.5)</pre>
	 * 
	 * <pre> -R 
	 *  No resampling. (default resampling is on)</pre>
	 * 
	 * <pre> -P 
	 *  Set the size of the subsample (fraction of the training set). (default 0.5)</pre>
	 *  
	 * <pre> -Q &lt;technique&gt;
	 *  Set the minimization technique:
	 *    0 = gradient descent,
	 *    1 = Newton-Raphson.
	 *  (default 0) </pre>
	 * 
	   <!-- options-end -->
	 *
	 * @param options the list of options as an array of strings
	 * @throws Exception if an option is not supported 
	 */
	public void setOptions(String[] options) throws Exception {
		
		String stringM = Utils.getOption('M', options);
		if (stringM.length() != 0)
			nRules = Integer.parseInt(stringM);
		
		String stringNu = Utils.getOption('S', options);
		if (stringNu.length() != 0)
			nu = Double.parseDouble(stringNu);
		
		resample = Utils.getFlag('R', options);
		
		String stringPercentage = Utils.getOption('P', options);
		if (stringPercentage.length() != 0)
			percentage = Double.parseDouble(stringPercentage);
		
		String stringTechnique = Utils.getOption('Q', options);
		if (stringTechnique.length() != 0)
			minimization = Integer.parseInt(stringTechnique);

	}

	/**
	 * Gets the current settings of the classifier.
	 *
	 * @return an array of strings suitable for passing to setOptions
	 */
	public String[] getOptions() {
		
		Vector<String> results = new Vector<String>();
		
		results.add("-M");
		results.add("" + nRules);
		
		results.add("-S");
		results.add("" + nu);
		
		if (!resample)
			results.add("-R");

		results.add("-P");
		results.add("" + percentage);
		
		results.add("-Q");
		results.add("" + minimization);
		
		return results.toArray(new String[results.size()]);	  
	}



	
	/**
	 * Classifies a given instance.
	 *
	 * @param instance the instance to be classified
	 * @return index of the predicted class
	 */
	public double classifyInstance(Instance instance) {
		double[] evalF = evaluateF(instance);
		int classIndex = 0;
		for (int k = 1; k < K; k++)
			if(evalF[classIndex] < evalF[k])
				classIndex = k;
		return classIndex;
	}
	
	public double[][] multipleEvaluateF(Instance instance) {
		double[][] evalsF = new double[nRules][K];
		ntb.input(instance); 
		instance = ntb.output();
		
		for (int i = 0; i < nRules; i++) {
			double[] currentValues = rules[i].classifyInstance(instance);
			if (currentValues != null) {
				for (int j = 0; j < K; j++) {
					if (i == 0)
						evalsF[i][j] = defaultRule[j] + currentValues[j];
					else
						evalsF[i][j] = evalsF[i-1][j] + currentValues[j];
				}
			}
			else {
				for (int j = 0; j < K; j++) {
					if (i == 0)
						evalsF[i][j] = defaultRule[j];
					else
						evalsF[i][j] = evalsF[i-1][j];
				}
			}
		}
		return evalsF;
	}

	
	public double[] multipleClassifyInstance(Instance instance) {
		
		double[][] values = multipleEvaluateF(instance);
		
		double[] curve = new double[nRules];
		
		for (int j = 0; j < nRules; j++) {
			int classIndex = 0;
			for (int k = 1; k < K; k++)
				if(values[j][classIndex] < values[j][k])
					classIndex = k;
			curve[j] = classIndex;
		}
		return curve;
	}
	
	
	public double computeEmpiricalRisk() {
		double empiricalRisk = 0;
		for (int i = 0; i < N; i++) {
			double total = 0;
			for (int k = 0; k < K; k++)
				total+= Math.exp(f[i][k]);
			empiricalRisk -= instances.instance(i).weight() * Math.log(Math.exp(f[i][(int)instances.instance(i).classValue()]) / total);
		}
		return empiricalRisk / N;
	}
		
	/**
	 * @param percentage the percentage to set
	 */
	public void setPercentage(double percentage) {
		this.percentage = percentage;
	}

	/**
	 * @return the percentage
	 */
	public double getPercentage() {
		return percentage;
	}

	/**
	 * @param nu the nu to set
	 */
	public void setNu(double nu) {
		this.nu = nu;
	}

	/**
	 * @return the nu
	 */
	public double getNu() {
		return nu;
	}

	public void setResample(boolean resample) {
		this.resample = resample;
	}

	public boolean getResample() {
		return resample;
	}

	/*
	<!-- Those values should not be changed by a user-->
	
	public void setPrechoiseK(boolean prechoiseK) {
		this.chooseClass = prechoiseK;
	}

	public boolean isPrechoiseK() {
		return chooseClass;
	}

	public void setR(double r) {
		R = r;
	}

	public double getR() {
		return R;
	}

	public void setRp(double rp) {
		Rp = rp;
	}

	public double getRp() {
		return Rp;
	}

	public void setUseLineSearch(boolean useLineSearch) {
		this.useLineSearch = useLineSearch;
	}

	public boolean isUseLineSearch() {
		return useLineSearch;
	}

	*/
	public static void main(String[] args) {
		runClassifier(new MLRules(), args);
	}
		
	public String nuTipText() {
		return "Shrinkage.";
	}

	public String resampleTipText() {
		return "Resampling";
	}

	public String percentageTipText() {
		return "Subsample size (as a fraction of the training set).";
	}

	public void setMinimization(SelectedTag newType) {

		if (newType.getTags() == TAGS_MINIMIZER) {
			minimization = newType.getSelectedTag().getID();
		}
	}
	
	public SelectedTag getMinimization() {

		return new SelectedTag(minimization, TAGS_MINIMIZER);
	}
	
	public String minimizationTipText() {
		return "Minimization technique.";
	}

}