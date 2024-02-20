package weka.classifiers.rules;

import java.util.Enumeration;
import java.util.Vector;

import pl.poznan.put.cs.idss.ml.empiricalRiskMinimizers.AbsoluteErrorRiskMinimizer;
import pl.poznan.put.cs.idss.ml.empiricalRiskMinimizers.EmpiricalRiskMinimizer;
import pl.poznan.put.cs.idss.ml.empiricalRiskMinimizers.GradientEmpiricalRiskMinimizer;
import pl.poznan.put.cs.idss.ml.empiricalRiskMinimizers.LeastAngleEmpiricalRiskMinimizer;
import pl.poznan.put.cs.idss.ml.ender.Regender;
import pl.poznan.put.cs.idss.ml.ender.RuleBuilder;
import pl.poznan.put.cs.idss.ml.ender.SingleRuleBuilder;
import pl.poznan.put.cs.idss.ml.lossFunctions.AbsoluteErrorLossFunction;
import pl.poznan.put.cs.idss.ml.lossFunctions.LossFunction;
import pl.poznan.put.cs.idss.ml.lossFunctions.SquaredLossFunction;
import weka.classifiers.Classifier;
import weka.classifiers.functions.supportVector.Kernel;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.SelectedTag;
import weka.core.Summarizable;
import weka.core.Tag;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.core.Capabilities.Capability;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;

/**
<!-- globalinfo-start -->
* Regression Ensemble of Decision Rules (RegENDER) - class for building an ensemble of regression rules.<br/> 
* Rules are combined in additive way.<br/>
* Can deal with both squared error and absolute error loss functions.<br/>";	    
* <br/>
* For more information, see:<br/>
* <br/>
* Krzysztof Dembczynski, Wojciech Kotlowski, Roman Slowinski:<br/>
* <i>Solving Regression by Learning an Ensemble of Decision Rules</i><br/>
* International Conference on Artificial Intelligence and Soft Computing 2008.<br/> 
* Lecture Notes in Artificial Intelligence, 5097 (2008), 144-151. Springer-Verlag.
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
  title     = {Solving Regression by Learning an Ensemble of Decision Rules},
  booktitle = {International Conference on Artificial Intelligence and
               Soft Computing, 2008},
  series    = {Lecture Notes in Artificial Intelligence},
  year      = {2008},
  volume    = {5097},
  pages     = {533--544},
  publisher = {Springer-Verlag}
}
* </pre>
* <p/>
<!-- technical-bibtex-end -->
*
<!-- options-start -->
* Valid options are: <p/>
* 
* <pre> -N &lt;number of rules&gt;
*  Set the number of rules, i.e. the ensemble size.
*  (default 50)</pre>
* 
* <pre> -S &lt;shrinkage&gt;
*  Set the amount of shrinkage.
*  (default 1.0)</pre>
* 
* <pre> -R 
*  No resampling. (default resampling is on)</pre>
* 
* <pre> -W 
*  Resampling is done with replacement. (default off)</pre>
*
* <pre> -P 
*  Set the size of the subsample (fraction of the training set). (default 0.5)</pre>
*
* <pre> -V
*  Replace missing values by means and modes.
*  (default off)</pre>
* 
* <pre> -Q &lt;technique&gt;
*  Set the minimization technique:
*    0 = simultaneous minimization,
*    1 = gradient descent.
*  (default 0) </pre>
* 
* <pre> -L &lt;loss&gt;
*  Set the loss function type:</pre>
*    0 = squared error loss,
*    1 = absolute error loss.
*  (default 0) </pre>
* 
<!-- options-end -->
*
* @author Krzysztof Dembczynski (kdembczynski@cs.put.poznan.pl)
* @author Wojciech Kotlowski (wkotlowski@cs.put.poznan.pl)
*/


public class RegENDER extends Classifier implements OptionHandler, TechnicalInformationHandler {
	
	public static int MINIMIZER_SIMULTANEOUS = 0;
	public static int MINIMIZER_GRADIENT_DESCENT = 1;
	
	public static int LOSS_SQUARED_ERROR = 0;
	public static int LOSS_ABSOLUTE_ERROR = 1;

	public static final Tag [] TAGS_MINIMIZER = {
		new Tag(MINIMIZER_SIMULTANEOUS, "Simultaneous minimizaiton"),
		new Tag(MINIMIZER_GRADIENT_DESCENT, "Gradient descent")
	};

	public static final Tag [] TAGS_LOSS = {
		new Tag(LOSS_SQUARED_ERROR, "Squared error loss"),
		new Tag(LOSS_ABSOLUTE_ERROR, "Absolute error loss")
	};

	private boolean modelBuilt = false;
	
	private Regender regender = null;
	
	/**
	 * Number of rules.
	 */
	private int M = 100;
	
	/**
	 * Shrinkage
	 */
	private double nu = 1;
	
	/**
	 * Resampling
	 */
	private boolean resample = true;
	
	/**
	 * Percentage (fraction) fo the training set used in each subsample
	 */
	private double percentage = 0.5;
	
	/**
	 * Resampling with replacement
	 */
	private boolean withReplacement = false;
	
	/**
	 * Replace missing values with means and modes
	 */
	private boolean replaceMissingValues = false;
	
	/**
	 * Minimization technique
	 */
	private int minimizationTechnique = MINIMIZER_SIMULTANEOUS;
	
	/**
	 * Loss function
	 */
	private int lossFunction = LOSS_SQUARED_ERROR;

	/** for serialization */
	static final long serialVersionUID = 23445541465867954L;
	
	
	
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
		
		result.setValue(Field.AUTHOR, "Krzysztof Dembczy{\\'n}ski and Wojciech Kot{\\l}owski and Roman S{\\l}owi{\\'n}ski");
		result.setValue(Field.TITLE, "Solving Regression by Learning an Ensemble of Decision Rules");
		result.setValue(Field.BOOKTITLE, "International Conference on Artificial Intelligence and Soft Computing, 2008");
		result.setValue(Field.SERIES, "Lecture Notes in Artificial Intelligence");
		result.setValue(Field.VOLUME, "5097");
		result.setValue(Field.YEAR, "2008");
		result.setValue(Field.PAGES, "533--544");
		result.setValue(Field.PUBLISHER, "Springer-Verlag");

		return result;
	}
	
	/**
	 * Returns a string describing classifier
	 * @return a description suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String globalInfo() {

		return "Regression Ensemble of Decision Rules (RegENDER) - class for building an ensemble of regression rules.\n" 
		+ "Rules are combined in additive way.\n"
		+ "Can deal with both squared error and absolute error loss functions.\n\n" 
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
	    result.enable(Capability.NUMERIC_CLASS);

		// instances
		result.setMinimumNumberInstances(1);

		return result;
	}	
	
	/**
	 * Returns a description of the classifier.
	 *
	 * @return a description of the classifier as a string.
	 */
	public String toString() {

		if (!modelBuilt) {
			return "Regression Ensemble of Decision Rules: No model built yet.";
		}
		else {
			StringBuffer buffer = new StringBuffer("Regression Ensemble of Decision Rules...\n\n" +
					                               regender.getM() + " rules generated.\n" +
					                               "Default rule: " + regender.getDefaultRule() + "\n\n" +
					                               "List of decision rules:\n\n");
			
			for (int i = 0; i < regender.getM(); i++)
				buffer.append(regender.getRules()[i].toString() + "\n");
			return buffer.toString();
		}
	}
	
	/**
	 * Classifies a given instance.
	 *
	 * @param instance the instance to be classified
	 * @return index of the predicted class
	 */
	public double classifyInstance(Instance instance) {

		return regender.classifyInstance(instance);
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
	    
	    
	    LossFunction loss = null;
	    
	    if (lossFunction == LOSS_ABSOLUTE_ERROR)
	    	loss = new AbsoluteErrorLossFunction();
	    else
	    	loss = new SquaredLossFunction();
	    
	    EmpiricalRiskMinimizer empiricalRiskMinimizer = null;
	    if (minimizationTechnique == MINIMIZER_GRADIENT_DESCENT)
	    	empiricalRiskMinimizer = new LeastAngleEmpiricalRiskMinimizer();
	    else {
	    	if (lossFunction == LOSS_ABSOLUTE_ERROR)
	    		empiricalRiskMinimizer = new AbsoluteErrorRiskMinimizer();
	    	else
	    		empiricalRiskMinimizer = new GradientEmpiricalRiskMinimizer();
	    }
	    
	    empiricalRiskMinimizer.setLossFunction(loss);
	    
	    SingleRuleBuilder ruleBuilder = new SingleRuleBuilder(empiricalRiskMinimizer);
	    ruleBuilder.setNu(nu);
	    
	    regender = new Regender(M, ruleBuilder, resample, percentage, withReplacement);
	    
	    regender.buildClassifier(instances);

	    modelBuilt = true;
		
	}

	/**
	 * Calculates the class membership probabilities for the given test instance.
	 *
	 * @param instance the instance to be classified
	 * @return predicted class probability distribution
	 * @throws Exception if class is numeric
	 */
	public double [] distributionForInstance(Instance instance) 
	throws Exception {

		return regender.distributionForInstance(instance);
	}
	
	/**
	 * Returns an enumeration describing the available options.
	 *
	 * @return an enumeration of all the available options.
	 */
	public Enumeration listOptions() {
		Vector result = new Vector();

		result.addElement(new Option(
				"\tSet the number of rules, i.e. the ensemble size (default 100).",
				"N", 1, "-N <number of rules>"));

		result.addElement(new Option(
				"\tSet the amount of shrinkage (default 1.0).",
				"S", 1, "-S <shrinkage>"));
		
		result.addElement(new Option(
				"\tNo resampling (default resampling is on).",
				"R", 0, "-R"));
		
		result.addElement(new Option(
				"\tResampling is done with replacement (default off).",
				"W", 0, "-W"));

		result.addElement(new Option(
				"\tReplace missing values by means and modes (default off).",
				"V", 0, "-V"));
		
		result.addElement(new Option(
				"\tSet the size of the subsample as a fraction of the training set (default 0.5).",
				"P", 1, "-P"));

		result.addElement(new Option(
				"\tSet the minimization technique:\n" 
				+ "\t\t0 = simultaneous minimization,\n"
				+ "\t\t1 = gradient descent.",
				"Q", 1, "-Q <technique>"));
		
		result.addElement(new Option(
				"\tSet the loss function:\n" 
				+ "\t\t0 = squared error loss,\n"
				+ "\t\t1 = absolute error loss.",
				"L", 1, "-L <loss>"));

		return result.elements();
	}

	/**
	 * Parses a given list of options. <p/>
	 *
	   <!-- options-start -->
	 * Valid options are: <p/>
	 * 
	 * <pre> -N &lt;number of rules&gt;
	 *  Set the number of rules, i.e. the ensemble size.
	 *  (default 100)</pre>
	 * 
	 * <pre> -S &lt;shrinkage&gt;
	 *  Set the amount of shrinkage.
	 *  (default 1.0)</pre>
	 * 
	 * <pre> -R 
	 *  No resampling. (default resampling is on)</pre>
	 * 
	 * <pre> -W 
	 *  Resampling is done with replacement. (default off)</pre>
	 *
	 * <pre> -P 
	 *  Set the size of the subsample (fraction of the training set). (default 0.5)</pre>
	 *  
	 * <pre> -V
	 *  Replace missing values by means and modes.
	 *  (default off)</pre>
	 * 
	 * <pre> -Q &lt;technique&gt;
	 *  Set the minimization technique:
	 *    0 = simultaneous minimization,
	 *    1 = gradient descent.
	 *  (default 0) </pre>
	 * 
	 * <pre> -L &lt;loss&gt;
	 *  Set the loss function type:</pre>
	 *    0 = squared error loss,
	 *    1 = absolute error loss.
	 *  (default 0) </pre>
	 * 
	   <!-- options-end -->
	 *
	 * @param options the list of options as an array of strings
	 * @throws Exception if an option is not supported 
	 */
	public void setOptions(String[] options) throws Exception {
		
		String stringM = Utils.getOption('N', options);
		if (stringM.length() != 0)
			M = Integer.parseInt(stringM);
		
		String stringNu = Utils.getOption('S', options);
		if (stringNu.length() != 0)
			nu = Double.parseDouble(stringNu);
		
		resample = Utils.getFlag('R', options);
		
		withReplacement = Utils.getFlag('W', options);
		
		replaceMissingValues = Utils.getFlag('V', options);
			
		String stringPercentage = Utils.getOption('P', options);
		if (stringPercentage.length() != 0)
			percentage = Double.parseDouble(stringPercentage);
		
		String stringTechnique = Utils.getOption('Q', options);
		if (stringTechnique.length() != 0)
			minimizationTechnique = Integer.parseInt(stringTechnique);

		String stringLoss = Utils.getOption('L', options);
		if (stringLoss.length() != 0)
			lossFunction = Integer.parseInt(stringLoss);
		
	}

	/**
	 * Gets the current settings of the classifier.
	 *
	 * @return an array of strings suitable for passing to setOptions
	 */
	public String[] getOptions() {
		
		Vector<String> results = new Vector<String>();
		
		results.add("-N");
		results.add("" + M);
		
		results.add("-S");
		results.add("" + nu);
		
		if (!resample)
			results.add("-R");
		
		if (withReplacement)
			results.add("-W");
		
		if (replaceMissingValues)
			results.add("-V");
		
		results.add("-P");
		results.add("" + percentage);
		
		results.add("-Q");
		results.add("" + minimizationTechnique);
		
		results.add("-L");
		results.add("" + lossFunction);
		
		return results.toArray(new String[results.size()]);	  
	}


	
	public static void main(String[] args) {	
		runClassifier(new RegENDER(), args);
	}

	public void setm(int m) {
		M = m;
	}

	public int getm() {
		return M;
	}
	
	public String MTipText() {
		return "Number of rules.";
	}

	public void setNu(double nu) {
		this.nu = nu;
	}

	public double getNu() {
		return nu;
	}
	
	public String nuTipText() {
		return "Shrinkage.";
	}

	public void setResample(boolean resample) {
		this.resample = resample;
	}

	public boolean getResample() {
		return resample;
	}

	public String resampleTipText() {
		return "Resampling";
	}

	public void setPercentage(double percentage) {
		this.percentage = percentage;
	}

	public double getPercentage() {
		return percentage;
	}
	
	public String percentageTipText() {
		return "Subsample size (as a fraction of the training set).";
	}

	public void setWithReplacement(boolean withReplacement) {
		this.withReplacement = withReplacement;
	}

	public boolean getWithReplacement() {
		return withReplacement;
	}

	public String withReplacementTipText() {
		return "Resampling with replacement.";
	}

	public void setReplaceMissingValues(boolean replaceMissingValues) {
		this.replaceMissingValues = replaceMissingValues;
	}

	public boolean getReplaceMissingValues() {
		return replaceMissingValues;
	}
	
	public String replaceMissingValuesTipText() {
		return "Replace missing values by means and modes.";
	}

	
	public void setMinimizationTechnique(SelectedTag newType) {

		if (newType.getTags() == TAGS_MINIMIZER) {
			minimizationTechnique = newType.getSelectedTag().getID();
		}
	}
	
	public SelectedTag getMinimizationTechnique() {

		return new SelectedTag(minimizationTechnique, TAGS_MINIMIZER);
	}
	
	public String minimizationTechniqueTipText() {
		return "Minimization technique.";
	}


	public void setLossFunction(SelectedTag newType) {

		if (newType.getTags() == TAGS_LOSS) {
			lossFunction = newType.getSelectedTag().getID();
		}
	}
	
	public SelectedTag getLossFunction() {
		return new SelectedTag(lossFunction, TAGS_LOSS);
	}
	
	public String lossFunctionTipText() {
		return "Loss function.";
	}
	

}
