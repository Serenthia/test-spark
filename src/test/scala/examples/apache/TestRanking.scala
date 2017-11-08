package examples.apache

import org.apache.spark.sql.SparkSession
import org.specs2.Specification

class TestRanking extends Specification {
  def is = s2"""
      Ranking
        can correctly parse input file $testParseInputFile
        can binarise input rankings $testBinariseRankings
    """

  val data: TestRankingData = new TestRankingData
  val context: SparkSession = data.context
  val ratings = SparkRanking.readInputData(context, "data/mllib/sample_movielens_data.txt")

  def testParseInputFile = {
    ratings.first().must_==(data.Outputs.inputFileFirstRow).and(
      SparkRanking.analyseRatings(context, ratings).must_==(data.Outputs.inputFileCharacteristics))
  }

  def testBinariseRankings = {
    SparkRanking.binariseRatings(context, ratings).collect.take(5).must_==(data.Outputs.binarisedRatings)
  }
}
