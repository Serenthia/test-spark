package examples.apache

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.{ Row, SparkSession }
import org.specs2.Specification

class TestLogisticRegression extends Specification {
  def is = s2"""
      Logistic Regression
        can train logistic regression model $testTrainLogisticRegressionModel
        can test logistic regression model $testTestLogisticRegressionModel
        obtain maximum f-measure $testMaxRegressionFMeasure
        fit using Classification example data $testExampleClassificationFitting
        obtain model info using Classification example data $testExampleClassificationDetails
  """

  val data: TestData = new TestData
  val context: SparkSession = data.context

  def testTrainLogisticRegressionModel = {
    val model = data.Inputs.trainedLogisticRegressionModel

    model.coefficientMatrix.toArray.must_==(data.Outputs.logisticRegressionModelCoefficients).and(
      model.intercept.must_==(data.Outputs.logisticRegressionModelIntercept))
  }

  def testTestLogisticRegressionModel = {
    val model = data.Inputs.testedLogisticRegressionModel.select("features", "label", "probability", "prediction")

    model.collect.foreach { case Row(features: Vector, label: Double, prob: Vector, prediction: Double) => (features, label, prob, prediction) }.must_==(
      data.Outputs.testedLogisticRegressionModel.collect.foreach { case Row(features: Vector, label: Double, prob: Vector, prediction: Double) => (features, label, prob, prediction) })
  }

  def testMaxRegressionFMeasure = {
    SparkLogisticRegression.maximiseFMeasure(context, data.Inputs.trainedLogisticRegressionModel).getThreshold.must_==(data.Outputs.logisticRegressionThreshold)
  }

  def testExampleClassificationFitting = {
    val model = data.Inputs.classificationFittedModel
    val multinomialModel = data.Inputs.classificationMultinomialModel

    model.coefficients.must_==(data.Outputs.classificationModelCoefficients).and(
      model.intercept.must_==(data.Outputs.classificationModelIntercept)).and(
        multinomialModel.coefficientMatrix.toArray.must_==(data.Outputs.classificationMultinomialCoefficients)).and(
          multinomialModel.interceptVector.must_==(data.Outputs.classificationMultinomialIntercepts))
  }

  def testExampleClassificationDetails = {
    val model = data.Inputs.classificationFittedModel

    SparkLogisticRegression.trainingSummary(model).objectiveHistory.must_==(data.Outputs.classificationModelSummary).and(
      SparkLogisticRegression.binarySummary(model).areaUnderROC.must_==(data.Outputs.classificationROCArea)).and(
        SparkLogisticRegression.binarySummary(model).roc.limit(5).collect().map(r => (r(0).toString.toDouble, r(1).toString.toDouble)).must_==(data.Outputs.classificationROC))
  }
}
