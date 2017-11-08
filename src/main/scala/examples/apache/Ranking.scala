package examples.apache

import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.sql.{ Dataset, SparkSession }

object SparkRanking {
  case class RatingsCharacteristics(numUsers: Long, numProducts: Long, numRatings: Long)

  def readInputData(context: SparkSession, filePath: String, userColIndex: Int = 0, productColIndex: Int = 1, ratingColIndex: Int = 2): Dataset[Rating] = {
    import context.implicits._

    val input = IO.fileToDataFrame(context, filePath)
    input.printSchema()

    input.map { line =>
      val fields = line.toString().split("::")
      println(fields)
      Rating(fields(userColIndex).toInt, fields(productColIndex).toInt, fields(ratingColIndex).toDouble - 2.5)
    }.cache
  }

  def analyseRatings(context: SparkSession, ratings: Dataset[Rating]): RatingsCharacteristics = {
    import context.implicits._

    RatingsCharacteristics(
      ratings.map(_.user).distinct.count,
      ratings.map(_.product).distinct.count,
      ratings.count)
  }

  def binariseRatings(context: SparkSession, ratings: Dataset[Rating]): Dataset[Rating] = {
    import context.implicits._

    ratings.map(r => Rating(r.user, r.product, if (r.rating > 0) 1.0 else 0.0)).cache
  }
}
