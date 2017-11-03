package examples.apache

import org.apache.spark.sql.SparkSession
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
        can test logistic regression model $testTestLogisticRegressionModel
        obtain maximum f-measure $testMaxRegressionFMeasure
        fit using Classification example data $testExampleClassificationFitting
        obtain model info using Classification example data $testExampleClassificationDetails
  """

  val data = new TestData
  val context: SparkSession = data.context

  def testPearsonCorrelation = {
    SparkCorrelation.correlate(context, data.Inputs.correlation, "pearson").toString.must_==(data.Outputs.pearson.toString)
  }

  def testSpearmanCorrelation = {
    SparkCorrelation.correlate(context, data.Inputs.correlation, "spearman").toString.must_==(data.Outputs.spearman.toString)
  }

  def testChiSquared = {
    SparkHypothesisTesting.chiSquareTest(context, data.Inputs.chiSquared).must_==(data.Outputs.chiSquared)
  }

  def testTrainLogisticRegressionModel = {
    val model = data.Inputs.trainedLogisticRegressionModel

    model.coefficientMatrix.toArray.must_==(data.Outputs.logisticRegressionModelCoefficients).and(
      model.intercept.must_==(data.Outputs.logisticRegressionModelIntercept))
  }

  def testTestLogisticRegressionModel = {
    data.Inputs.testedLogisticRegressionModel.toString.must_==(data.Outputs.logisticRegressionModelCoefficients)
  }

  def testMaxRegressionFMeasure = {
    SparkLogisticRegression.maximiseFMeasure(context, data.Inputs.trainedLogisticRegressionModel).getThreshold.must_==(data.Outputs.logisticRegressionThreshold)
  }

  def testParamValidation = {
    /*    Param.ElasticNet(1.3).must(throwA[IllegalArgumentException]("Elastic net parameter must be between 0 and 1 inclusive")).and(
      Param.ElasticNet(-0.3).must(throwA[IllegalArgumentException]("Elastic net parameter must be between 0 and 1 inclusive"))).and(
        Param.ElasticNet(0.3).must(not(throwA[IllegalArgumentException]))).and(
          Param.MaxIterations(-3).must(throwA[IllegalArgumentException]("Max number of iterations must be greater than 0"))).and(
            Param.MaxIterations(3).must(not(throwA[IllegalArgumentException]))).and(
              Param.ClassificationThreshold(1.3).must(throwA[IllegalArgumentException]("Binary classification threshold must be between 0 and 1 inclusive"))).and(
                Param.ClassificationThreshold(-0.3).must(throwA[IllegalArgumentException]("Binary classification threshold must be between 0 and 1 inclusive"))).and(
                  Param.ClassificationThreshold(0.3).must(not(throwA[IllegalArgumentException])))*/
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
