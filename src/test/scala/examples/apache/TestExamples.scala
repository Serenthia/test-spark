package examples.apache

import org.apache.spark.ml.linalg.{Vectors, Vector}
import org.specs2.Specification

class TestExamples extends Specification {
  def is = s2"""
      Correlations
        can use Pearson method $testPearsonCorrelation
        can use Spearman method $testSpearmanCorrelation

      Hypothesis Testing
        can use Chi-squared method $testChiSquared
  """

  val context = new Context().spark
  val correlationInput = Seq(
    Vectors.sparse(4, Seq((0, 1.0), (3, -2.0))),
    Vectors.dense(4.0, 5.0, 0.0, 3.0),
    Vectors.dense(6.0, 7.0, 0.0, 8.0),
    Vectors.sparse(4, Seq((0, 9.0), (3, 1.0)))
  )

  def testPearsonCorrelation = {
    ExCorrelation.correlate(context, correlationInput, "pearson") must_== "NOT THIS"
  }

  def testSpearmanCorrelation = {
    ExCorrelation.correlate(context, correlationInput, "spearman") must_== "NOT THIS"
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
}
