package examples.apache

import org.apache.spark.ml.classification.{BinaryLogisticRegressionSummary, LogisticRegression}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.specs2.Specification

class TestExamples extends Specification {
  def is = s2"""
      Correlations
        can use Pearson method $testPearsonCorrelation
        can use Spearman method $testSpearmanCorrelation

      Hypothesis Testing
        can use Chi-squared method $testChiSquared

      Logistic Regression
        can train logistic regression model $testTrainLogisticRegressionModel
        can validate parameters $testParamValidation
        fit using Classification example data $testExampleClassificationFitting
        obtain model info using Classification example data $testExampleClassificationDetails
  """

  val context = new Context().spark
  val correlationInput = Seq(
    Vectors.sparse(4, Seq((0, 1.0), (3, -2.0))),
    Vectors.dense(4.0, 5.0, 0.0, 3.0),
    Vectors.dense(6.0, 7.0, 0.0, 8.0),
    Vectors.sparse(4, Seq((0, 9.0), (3, 1.0)))
  )

  def testPearsonCorrelation = {
    ExCorrelation.correlate(context, correlationInput, "pearson").must_==("NOT THIS")
  }

  def testSpearmanCorrelation = {
    ExCorrelation.correlate(context, correlationInput, "spearman").must_==("NOT THIS")
  }

  def testChiSquared = {
    ExHypothesisTesting.chiSquareTest(context, Seq(
      (0.0, Vectors.dense(0.5, 10.0)),
      (0.0, Vectors.dense(1.5, 20.0)),
      (1.0, Vectors.dense(1.5, 30.0)),
      (0.0, Vectors.dense(3.5, 30.0)),
      (0.0, Vectors.dense(3.5, 40.0)),
      (1.0, Vectors.dense(3.5, 40.0))
    )) must_== "NOT THIS"
  }

  def testTrainLogisticRegressionModel = {
    val model = ExLogisticRegression.trainModel(new LogisticRegression, context.createDataFrame(Seq(
      (1.0, Vectors.dense(0.0, 1.1, 0.1)),
      (0.0, Vectors.dense(2.0, 1.0, -1.0)),
      (0.0, Vectors.dense(2.0, 1.3, 1.0)),
      (1.0, Vectors.dense(0.0, 1.2, -0.5))
    )).toDF("label", "features"))

    model.must_==("NOT THIS")
  }

  def testParamValidation = {
    ParamElasticNet(1.3).must(throwA[IllegalArgumentException]("Elastic net parameter must be between 0 and 1 inclusive")).and(
      ParamElasticNet(-0.3).must(throwA[IllegalArgumentException]("Elastic net parameter must be between 0 and 1 inclusive"))
    )
  }

  def testExampleClassificationFitting = {
    val regressionTrainingData = context.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
    val model = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)
    val fittedModel = SparkLogisticRegression.trainModel(model, regressionTrainingData)

    val multinomialModel = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8).setFamily("multinomial")
    val fittedMultinomialModel = SparkLogisticRegression.trainModel(multinomialModel, regressionTrainingData)
    (fittedModel.coefficients, fittedModel.intercept).must_==((0, 0)).and(
      (fittedMultinomialModel.coefficientMatrix, fittedMultinomialModel.interceptVector).must_==((0, 0))
    )
  }

  def testExampleClassificationDetails = {
    val regressionTrainingData = context.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
    val model = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)
    val fittedModel = SparkLogisticRegression.trainModel(model, regressionTrainingData)
    val trainingSummary = fittedModel.summary
    val binarySummary = trainingSummary.asInstanceOf[BinaryLogisticRegressionSummary]
    val receiverOperatingCharacteristic = binarySummary.roc

    trainingSummary.objectiveHistory.must_==(List()).and(
      binarySummary.areaUnderROC.must_==("Area under ROC")
    )
  }
}
