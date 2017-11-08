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

  // This is a Transformer (produced by training the Estimator)
  def trainModel(estimator: LogisticRegression, trainingSet: DataFrame, maxIterations: Int, regularisation: Double = 0.3, elasticNet: Double = 0.8, family: String = "auto"): LogisticRegressionModel = {
    estimator
      .setMaxIter(maxIterations)
      .setRegParam(regularisation)
      .setElasticNetParam(elasticNet)
      .setFamily(family)
      .fit(trainingSet)
  }

  def testModel(transformer: LogisticRegressionModel, testSet: DataFrame): DataFrame = {
    transformer.transform(testSet).select("features", "label", "probability", "prediction")
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
