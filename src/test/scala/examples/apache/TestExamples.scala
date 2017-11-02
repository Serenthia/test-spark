package examples.apache

import org.apache.spark.ml.classification.{BinaryLogisticRegressionSummary, LogisticRegression}
import org.specs2.Specification
import examples.apache.TestData._

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

  def testPearsonCorrelation = {
    ExCorrelation.correlate(context, Inputs.correlation, "pearson").toString.must_==(Outputs.pearson.toString)
  }

  def testSpearmanCorrelation = {
    ExCorrelation.correlate(context, Inputs.correlation, "spearman").toString.must_==(Outputs.spearman.toString)
  }

  def testChiSquared = {
    ExHypothesisTesting.chiSquareTest(context, Inputs.chiSquared).must_==(Outputs.chiSquared)
  }

  def testTrainLogisticRegressionModel = {
    Inputs.logisticRegressionModel.coefficientMatrix.toArray.must_==(Outputs.logisticRegressionModelCoefficients).and(
      Inputs.logisticRegressionModel.intercept.must_==(Outputs.logisticRegressionModelIntercept)
    )
  }

  def testParamValidation = {
    ParamElasticNet(1.3).must(throwA[IllegalArgumentException]("Elastic net parameter must be between 0 and 1 inclusive")).and(
      ParamElasticNet(-0.3).must(throwA[IllegalArgumentException]("Elastic net parameter must be between 0 and 1 inclusive")).and(
        ParamElasticNet(0.5).must(not(throwA[IllegalArgumentException]("Elastic net parameter must be between 0 and 1 inclusive")))
      )
    )
  }

  def testExampleClassificationFitting = {
    Inputs.classificationFittedModel.coefficients.must_==(Outputs.classificationModelCoefficients).and(
      Inputs.classificationFittedModel.intercept.must_==(Outputs.classificationModelIntercept)
    ).and(
      Inputs.classificationMultinomialModel.coefficientMatrix.toArray.must_==(Outputs.classificationMultinomialCoefficients)
    ).and(
      Inputs.classificationMultinomialModel.interceptVector.must_==(Outputs.classificationMultinomialIntercepts)
    )
  }

  def testExampleClassificationDetails = {
    Inputs.classificationModelSummmary.objectiveHistory.must_==(Outputs.classificationModelSummary).and(
      Inputs.classificationModelBinarySummary.areaUnderROC.must_==(Outputs.classificationROCArea)
    )
  }
}
