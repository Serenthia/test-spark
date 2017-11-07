package examples.apache

import org.apache.spark.sql.SparkSession

class Context() {
  val spark: SparkSession = SparkSession.builder().appName("DocumentationTest").master("local").getOrCreate()
}
