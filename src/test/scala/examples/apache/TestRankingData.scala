package examples.apache

import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.sql.{ DataFrame, Dataset, SparkSession }

class TestRankingData {
  val context: SparkSession = new Context().spark

  val rankingTrainingData: DataFrame = context.read.format("libsvm").load("data/mllib/sample_movielens_data.txt")

  object Inputs {
  }

  object Outputs {
    val inputFileFirstRow: Rating = {
      Rating(0, 0, 1.2)
    }

    val inputFileCharacteristics: Map[String, Int] = {
      Map(
        "ratings" -> 10,
        "users" -> 10,
        "movies" -> 10)
    }

    val binarisedRatings: Array[Rating] = Array.empty
  }
}
