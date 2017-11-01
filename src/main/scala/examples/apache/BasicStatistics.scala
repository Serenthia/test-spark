package examples.apache

import org.apache.spark.ml.linalg.{Matrix, Vector}
import org.apache.spark.ml.stat.{ChiSquareTest, Correlation}
import org.apache.spark.sql.{Row, SparkSession}

class Context() {
  val spark = SparkSession.builder ().appName ("DocumentationTest").master ("local").getOrCreate ()
}

case class ChiSquareResults (
  pValues: Vector,
  degreesOfFreedom: String,
  statistics: Vector
)

object ExCorrelation {
  def correlate(context: SparkSession, data: Seq[Vector], method: String): Matrix = {
    import context.implicits._
    val df = data.map(Tuple1.apply).toDF("features")
    val Row(coefficient: Matrix) = Correlation.corr(df, "features", method).head
    coefficient
  }
}

object ExHypothesisTesting {
  def chiSquareTest(context: SparkSession, data: Seq[(Double, Vector)]): ChiSquareResults = {
    import context.implicits._
    val df = data.toDF("label", "features")
    val chi = ChiSquareTest.test(df, "features", "label").head
    ChiSquareResults(chi.getAs[Vector](0), chi.getSeq[Int](1).mkString("[", ",", "]"), chi.getAs[Vector](2))
  }
}
