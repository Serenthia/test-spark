package examples.apache

import org.apache.spark.ml.classification.{BinaryLogisticRegressionSummary, LogisticRegression, LogisticRegressionModel, LogisticRegressionTrainingSummary}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.functions.max

object SparkLogisticRegression {
  def withParams(estimator: LogisticRegression, params: ParamMap): LogisticRegression = {
    params.toSeq.map(p => estimator.set(p.param.asInstanceOf, p.value))
    estimator
  }

  // This is a Transformer (produced by training the Estimator)
  def trainModel(estimator: LogisticRegression, trainingSet: DataFrame): LogisticRegressionModel = estimator.fit(trainingSet)

  def testModel(transformer: LogisticRegressionModel, testSet: DataFrame) = {
    transformer.transform(testSet).select("features", "label", "myProbability", "prediction").collect().foreach {
      case Row(features: Vector, label: Double, prob: Vector, prediction: Double) =>
        ModelTestResults(features, label, prob, prediction)
    }
    transformer
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

  def maximiseFMeasure(context: SparkSession, trainedModel: LogisticRegressionModel) = {
    import context.implicits._
    val fMeasure = binarySummary(trainedModel).fMeasureByThreshold
    val maxFMeasure = fMeasure.select(max("F-Measure")).head().getDouble(0)
    val bestThreshold = fMeasure.where($"F-Measure" === maxFMeasure).select("threshold").head().getDouble(0)
    trainedModel.setThreshold(bestThreshold)
  }
}
