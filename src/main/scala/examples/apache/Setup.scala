package examples.apache

import org.apache.spark.sql.{ DataFrame, SparkSession }

class Context() {
  val spark: SparkSession = SparkSession.builder().appName("DocumentationTest").master("local").getOrCreate()
}

object IO {
  def fileToDataFrame(context: SparkSession, filePath: String): DataFrame = {
    context.read.format("libsvm").load(filePath)
  }
}
