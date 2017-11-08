package examples.apache

import org.apache.spark.sql.SparkSession
import org.specs2.Specification

class TestBasicStatistics extends Specification {
  def is = s2"""
      Correlations
        can use Pearson method $testPearsonCorrelation
        can use Spearman method $testSpearmanCorrelation

      Hypothesis Testing
        can use Chi-squared method $testChiSquared
  """

  val data: TestData = new TestData
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
}