package examples.apache

import org.apache.spark.ml.classification.{BinaryLogisticRegressionSummary, LogisticRegression, LogisticRegressionModel, LogisticRegressionTrainingSummary}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.functions.max
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.{DataFrame, Row}

case class ModelTestResults (
  features: Vector,
  label: Double,
  probability: Vector,
  prediction: Double
)

sealed trait SparkParam {
}
case class ParamElasticNet (value: Double = 0.0) extends SparkParam {
  require(value >= 0 & value <= 1, "Elastic net parameter must be between 0 and 1 inclusive")
}
case class ParamFitIntercept (value: Boolean) extends SparkParam
case class ParamMaxIterations (value: Int = 100) extends SparkParam {
  require(value > 0, "Max number of iterations must be greater than 0")
}
case class ParamRegularisation (value: Double = 0.0) extends SparkParam
case class ParamStandardise (value: Boolean) extends SparkParam
case class ClassificationThreshold (value: Double) extends SparkParam {
  require(value >= 0 & value <= 1, "Binary classification threshold must be between 0 and 1 inclusive")
}
case class ParamConvergenceTolerance (value: Double) extends SparkParam

object ExLogisticRegression {

}




/*
aggregationDepth: suggested depth for treeAggregate (>= 2) (default: 2)
elasticNetParam: the ElasticNet mixing parameter, in range [0, 1]. For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty (default: 0.0)
family: The name of family which is a description of the label distribution to be used in the model. Supported options: auto, binomial, multinomial. (default: auto)
featuresCol: features column name (default: features)
fitIntercept: whether to fit an intercept term (default: true)
labelCol: label column name (default: label)
lowerBoundsOnCoefficients: The lower bounds on coefficients if fitting under bound constrained optimization. (undefined)
lowerBoundsOnIntercepts: The lower bounds on intercepts if fitting under bound constrained optimization. (undefined)
maxIter: maximum number of iterations (>= 0) (default: 100)
predictionCol: prediction column name (default: prediction)
probabilityCol: Column name for predicted class conditional probabilities. Note: Not all models output well-calibrated probability estimates! These probabilities should be treated as confidences, not precise probabilities (default: probability)
rawPredictionCol: raw prediction (a.k.a. confidence) column name (default: rawPrediction)
regParam: regularization parameter (>= 0) (default: 0.0)
standardization: whether to standardize the training features before fitting the model (default: true)
threshold: threshold in binary classification prediction, in range [0, 1] (default: 0.5)
thresholds: Thresholds in multi-class classification to adjust the probability of predicting each class. Array must have length equal to the number of classes, with values > 0 excepting that at most one value may be 0. The class with largest value p/t is predicted, where p is the original probability of that class and t is the class's threshold (undefined)
tol: the convergence tolerance for iterative algorithms (>= 0) (default: 1.0E-6)
upperBoundsOnCoefficients: The upper bounds on coefficients if fitting under bound constrained optimization. (undefined)
upperBoundsOnIntercepts: The upper bounds on intercepts if fitting under bound constrained optimization. (undefined)
weightCol: weight column name. If this is not set or empty, we treat all instance weights as 1.0 (undefined)
 */





/*

  // Prepare training data from a list of (label, features) tuples.
  /* val training = spark.createDataFrame(Seq(
    (1.0, Vectors.dense(0.0, 1.1, 0.1)),
    (0.0, Vectors.dense(2.0, 1.0, -1.0)),
    (0.0, Vectors.dense(2.0, 1.3, 1.0)),
    (1.0, Vectors.dense(0.0, 1.2, -0.5))
  )).toDF("label", "features")
  */


  // Prepare test data.
  /* val test = spark.createDataFrame(Seq(
    (1.0, Vectors.dense(-1.0, 1.5, 1.3)),
    (0.0, Vectors.dense(3.0, 2.0, -0.1)),
    (1.0, Vectors.dense(0.0, 2.2, -1.5))
  )).toDF("label", "features")
  */


  // Create a LogisticRegression instance. This instance is an Estimator.

  // Print out the parameters, documentation, and any default values.
  println("LogisticRegression parameters:\n" + lr.explainParams() + "\n")

  // We may set parameters using setter methods.



  // Since model1 is a Model (i.e., a Transformer produced by an Estimator),
  // we can view the parameters it used during fit().
  // This prints the parameter (name: value) pairs, where names are unique IDs for this
  // LogisticRegression instance.
  println("Model 1 was fit using parameters: " + model1.parent.extractParamMap)

  // We may alternatively specify parameters using a ParamMap,
  // which supports several methods for specifying parameters.
  val paramMap = ParamMap(lr.maxIter -> 20)
    .put(lr.maxIter, 30)  // Specify 1 Param. This overwrites the original maxIter.
    .put(lr.regParam -> 0.1, lr.threshold -> 0.55)  // Specify multiple Params.

  // One can also combine ParamMaps.
  val paramMap2 = ParamMap(lr.probabilityCol -> "myProbability")  // Change output column name.
  val paramMapCombined = paramMap ++ paramMap2

  // Now learn a new model using the paramMapCombined parameters.
  // paramMapCombined overrides all parameters set earlier via lr.set* methods.
  val model2 = lr.fit(training, paramMapCombined)
  println("Model 2 was fit using parameters: " + model2.parent.extractParamMap)



  // Make predictions on test data using the Transformer.transform() method.
  // LogisticRegression.transform will only use the 'features' column.
  // Note that model2.transform() outputs a 'myProbability' column instead of the usual
  // 'probability' column since we renamed the lr.probabilityCol parameter previously.
*/
