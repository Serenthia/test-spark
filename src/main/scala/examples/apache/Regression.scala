package examples.apache

import org.apache.spark.ml.classification.{ BinaryLogisticRegressionSummary, LogisticRegression, LogisticRegressionModel, LogisticRegressionTrainingSummary }
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.{ DataFrame, Row, SparkSession }
import org.apache.spark.sql.functions.max

object SparkLogisticRegression {
  case class ModelTestResults(
    label: Double,
    features: Vector,
    probability: Vector,
    prediction: Double)

  /*def createParamMap(params: List[SparkParam]): ParamMap = {
    val paramMap = new ParamMap()
    params.map(p => paramMap.put(p.paramPair))
    paramMap
  }*/

  // This is a Transformer (produced by training the Estimator)
  def trainModel(estimator: LogisticRegression, trainingSet: DataFrame, maxIterations: Int, regularisation: Double = 0.5, elasticNet: Double = 0.5, family: String = "auto"): LogisticRegressionModel = {
    /*val paramMap = createParamMap(List(
      MaxIterations(maxIterations),
      Regularisation(regularisation),
      ElasticNet(elasticNet)))*/
    estimator
      .fit(trainingSet)
      .setMaxIter(maxIterations)
      .setRegParam(regularisation)
      .setElasticNetParam(elasticNet)
      .setFamily(family)
  }

  def testModel(transformer: LogisticRegressionModel, testSet: DataFrame): DataFrame = {
    transformer.transform(testSet).select("features", "label", "probability", "prediction") /*.collect().foreach {
      case Row(features: Vector, label: Double, prob: Vector, prediction: Double) =>
        ModelTestResults(label, features, prob, prediction)
    }*/
  }

  def trainingSummary(trainedModel: LogisticRegressionModel): LogisticRegressionTrainingSummary = {
    trainedModel.summary
  }

  def binarySummary(trainedModel: LogisticRegressionModel): BinaryLogisticRegressionSummary = {
    trainingSummary(trainedModel).asInstanceOf[BinaryLogisticRegressionSummary]
  }

  def receiverOperatingCharacteristic(trainedModel: LogisticRegressionModel): DataFrame = {
    binarySummary(trainedModel).roc
  }

  def maximiseFMeasure(context: SparkSession, trainedModel: LogisticRegressionModel): LogisticRegressionModel = {
    import context.implicits._
    val fMeasure = binarySummary(trainedModel).fMeasureByThreshold
    val maxFMeasure = fMeasure.select(max("F-Measure")).head().getDouble(0)
    val bestThreshold = fMeasure.where($"F-Measure" === maxFMeasure).select("threshold").head().getDouble(0)
    trainedModel.setThreshold(bestThreshold)
  }
}

object SparkDecisionTree {

}
