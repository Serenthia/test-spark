package examples.apache

import org.apache.spark.ml.param._
import org.apache.spark.sql.SparkSession

class Context() {
  val spark: SparkSession = SparkSession.builder().appName("DocumentationTest").master("local").getOrCreate()
}

object Param {
  /*sealed trait SparkParam[T] {
    val value: T
    val paramPair: ParamPair[T]
  }

  case class ElasticNet(value: Double = 0.0) extends SparkParam[Double] {
    val paramPair: ParamPair[Double] = ParamPair(Param[Double], value)
    require(value >= 0 & value <= 1, "Elastic net parameter must be between 0 and 1 inclusive")
  }
  case class FitIntercept(value: Boolean) extends SparkParam[Boolean] {
    val paramPair = ParamPair(Param[Boolean], value)
  }
  case class MaxIterations(value: Int = 100) extends SparkParam[Int] {
    val paramPair = ParamPair(Param[Int], value)
    require(value > 0, "Max number of iterations must be greater than 0")
  }
  case class Regularisation(value: Double = 0.0) extends SparkParam[Double] {
    val paramPair = ParamPair(Param[Double], value)
  }
  case class Standardise(value: Boolean) extends SparkParam[Boolean] {
    val paramPair = ParamPair(Param[Boolean], value)
  }
  case class ClassificationThreshold(value: Double = 0.5) extends SparkParam[Double] {
    val paramPair = ParamPair(Param[Double], value)
    require(value >= 0 & value <= 1, "Binary classification threshold must be between 0 and 1 inclusive")
  }
  case class ConvergenceTolerance(value: Double = 1.0E-6) extends SparkParam[Double] {
    val paramPair = ParamPair(Param[Double], value)
  }
  case class LogisticRegressionFamily(value: String = "auto") extends SparkParam[String] {
    val paramPair = ParamPair(Param[String], value)
    require(List("auto", "binomial", "multinomial").contains(value), "Family must be 'auto', 'binomial' or 'multinomial'")
  }*/
}
