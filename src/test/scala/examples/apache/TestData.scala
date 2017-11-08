package examples.apache

import breeze.linalg.CSCMatrix
import org.apache.spark.ml.classification.{ LogisticRegression, LogisticRegressionModel }
import org.apache.spark.ml.linalg.{ Matrices, Matrix, Vector, Vectors }
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.sql.{ DataFrame, Dataset, SparkSession }

class TestData {
  val context: SparkSession = new Context().spark

  val logisticRegressionTrainData: DataFrame = {
    context.createDataFrame(Seq(
      (1.0, Vectors.dense(0.0, 1.1, 0.1)),
      (0.0, Vectors.dense(2.0, 1.0, -1.0)),
      (0.0, Vectors.dense(2.0, 1.3, 1.0)),
      (1.0, Vectors.dense(0.0, 1.2, -0.5)))).toDF("label", "features")
  }

  val logisticRegressionTestData: DataFrame = {
    context.createDataFrame(Seq(
      (1.0, Vectors.dense(-1.0, 1.5, 1.3)),
      (0.0, Vectors.dense(3.0, 2.0, -0.1)),
      (1.0, Vectors.dense(0.0, 2.2, -1.5)))).toDF("label", "features")
  }

  val classificationTrainingData: DataFrame = context.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
  val rankingTrainingData: DataFrame = context.read.format("libsvm").load("data/mllib/sample_movielens_data.txt")

  object Inputs {
    val correlation: Seq[Vector] = {
      Seq(
        Vectors.sparse(4, Seq((0, 1.0), (3, -2.0))),
        Vectors.dense(4.0, 5.0, 0.0, 3.0),
        Vectors.dense(6.0, 7.0, 0.0, 8.0),
        Vectors.sparse(4, Seq((0, 9.0), (3, 1.0))))
    }

    val chiSquared: Seq[(Double, Vector)] = {
      Seq(
        (0.0, Vectors.dense(0.5, 10.0)),
        (0.0, Vectors.dense(1.5, 20.0)),
        (1.0, Vectors.dense(1.5, 30.0)),
        (0.0, Vectors.dense(3.5, 30.0)),
        (0.0, Vectors.dense(3.5, 40.0)),
        (1.0, Vectors.dense(3.5, 40.0)))
    }

    val logisticRegression: LogisticRegression = new LogisticRegression

    val trainedLogisticRegressionModel: LogisticRegressionModel = {
      SparkLogisticRegression.trainModel(logisticRegression, logisticRegressionTrainData, 10)
    }

    val testedLogisticRegressionModel: DataFrame = {
      SparkLogisticRegression.testModel(trainedLogisticRegressionModel, logisticRegressionTestData)
    }

    val classificationFittedModel: LogisticRegressionModel = {
      SparkLogisticRegression.trainModel(logisticRegression, classificationTrainingData, 10)
    }

    val classificationMultinomialModel: LogisticRegressionModel = {
      SparkLogisticRegression.trainModel(logisticRegression, classificationTrainingData, 10, 0.3, 0.8, "multinomial")
    }

    val ratings: Dataset[Rating] = {
      SparkRanking.readInputData(context, "data/mllib/sample_movielens_data.txt")
    }

    val ratingsCharacteristics: SparkRanking.RatingsCharacteristics = {
      SparkRanking.analyseRatings(context, ratings)
    }

    val binarisedRatings: Array[Rating] = {
      SparkRanking.binariseRatings(context, ratings).collect.take(5)
    }
  }

  object Outputs {
    val pearson: Matrix = {
      Matrices.dense(4, 4,
        Array(
          1.0, 0.055641488407465814, Double.NaN, 0.4004714203168137,
          0.055641488407465814, 1.0, Double.NaN, 0.9135958615342522,
          Double.NaN, Double.NaN, 1.0, Double.NaN,
          0.4004714203168137, 0.9135958615342522, Double.NaN, 1.0))
    }

    val spearman: Matrix = {
      Matrices.dense(4, 4,
        Array(
          1.0, 0.10540925533894532, Double.NaN, 0.40000000000000174,
          0.10540925533894532, 1.0, Double.NaN, 0.9486832980505141,
          Double.NaN, Double.NaN, 1.0, Double.NaN,
          0.40000000000000174, 0.9486832980505141, Double.NaN, 1.0))
    }

    val chiSquared: ChiSquareResults = {
      ChiSquareResults(
        Vectors.dense(0.6872892787909721, 0.6822703303362126),
        "[2,3]",
        Vectors.dense(0.75, 1.5))
    }

    val testedLogisticRegressionModel: DataFrame = context.createDataFrame(Seq(
      (Vectors.dense(-1.0, 1.5, 1.3), 1.0, Vectors.dense(0.2523824340381209, 0.7476175659618791), 0.5))).toDF("features", "label", "probability", "prediction")

    val logisticRegressionModelCoefficients: Array[Double] = Array(-0.6108568753771977, -0.0, 0.0)

    val logisticRegressionModelIntercept: Double = 0.47508916388274547

    val logisticRegressionThreshold: Double = 0.6165875837363588

    val classificationModelCoefficients: Vector = Vectors.sparse(
      692,
      Array(244, 263, 272, 300, 301, 328, 350, 351, 378, 379, 405, 406, 407, 428, 433, 434, 455, 456, 461, 462, 483, 484, 489, 490, 496, 511, 512, 517, 539, 540, 568),
      Array(
        -7.353983524188197E-5, -9.102738505589466E-5, -1.9467430546904298E-4, -2.0300642473486668E-4, -3.1476183314863995E-5, -6.842977602660743E-5, 1.5883626898239883E-5,
        1.4023497091372047E-5, 3.5432047524968605E-4, 1.1443272898171087E-4, 1.0016712383666666E-4, 6.014109303795481E-4, 2.840248179122762E-4, -1.1541084736508837E-4,
        3.85996886312906E-4, 6.35019557424107E-4, -1.1506412384575676E-4, -1.5271865864986808E-4, 2.804933808994214E-4, 6.070117471191634E-4, -2.008459663247437E-4,
        -1.421075579290126E-4, 2.739010341160883E-4, 2.7730456244968115E-4, -9.838027027269332E-5, -3.808522443517704E-4, -2.5315198008555033E-4, 2.7747714770754307E-4,
        -2.443619763919199E-4, -0.0015394744687597765, -2.3073328411331293E-4))

    val classificationModelIntercept: Double = 0.22456315961250325

    val classificationMultinomialCoefficients: Array[Double] = {
      val matrixBuilder = new CSCMatrix.Builder[Double](2, 692)
      matrixBuilder.add(0, 244, 4.290365458958277E-5)
      matrixBuilder.add(1, 244, -4.290365458958294E-5)
      matrixBuilder.add(0, 263, 6.488313287833108E-5)
      matrixBuilder.add(1, 263, -6.488313287833092E-5)
      matrixBuilder.add(0, 272, 1.2140666790834663E-4)
      matrixBuilder.add(1, 272, -1.2140666790834657E-4)
      matrixBuilder.add(0, 300, 1.3231861518665612E-4)
      matrixBuilder.add(1, 300, -1.3231861518665607E-4)
      matrixBuilder.add(0, 350, -6.775444746760509E-7)
      matrixBuilder.add(1, 350, 6.775444746761932E-7)
      matrixBuilder.add(0, 351, -4.899237909429297E-7)
      matrixBuilder.add(1, 351, 4.899237909430322E-7)
      matrixBuilder.add(0, 378, -3.5812102770679596E-5)
      matrixBuilder.add(1, 378, 3.581210277067968E-5)
      matrixBuilder.add(0, 379, -2.3539704331222065E-5)
      matrixBuilder.add(1, 379, 2.353970433122204E-5)
      matrixBuilder.add(0, 405, -1.90295199030314E-5)
      matrixBuilder.add(1, 405, 1.90295199030314E-5)
      matrixBuilder.add(0, 406, -5.626696935778909E-4)
      matrixBuilder.add(1, 406, 5.626696935778912E-4)
      matrixBuilder.add(0, 407, -5.121519619099504E-5)
      matrixBuilder.add(1, 407, 5.1215196190995074E-5)
      matrixBuilder.add(0, 428, 8.080614545413342E-5)
      matrixBuilder.add(1, 428, -8.080614545413331E-5)
      matrixBuilder.add(0, 433, -4.256734915330487E-5)
      matrixBuilder.add(1, 433, 4.256734915330495E-5)
      matrixBuilder.add(0, 434, -7.080191510151425E-4)
      matrixBuilder.add(1, 434, 7.080191510151435E-4)
      matrixBuilder.add(0, 455, 8.094482475733589E-5)
      matrixBuilder.add(1, 455, -8.094482475733582E-5)
      matrixBuilder.add(0, 456, 1.0433687128309833E-4)
      matrixBuilder.add(1, 456, -1.0433687128309814E-4)
      matrixBuilder.add(0, 461, -5.4466605046259246E-5)
      matrixBuilder.add(1, 461, 5.4466605046259286E-5)
      matrixBuilder.add(0, 462, -5.667133061990392E-4)
      matrixBuilder.add(1, 462, 5.667133061990392E-4)
      matrixBuilder.add(0, 483, 1.2495896045528374E-4)
      matrixBuilder.add(1, 483, -1.249589604552838E-4)
      matrixBuilder.add(0, 484, 9.810519424784944E-5)
      matrixBuilder.add(1, 484, -9.810519424784941E-5)
      matrixBuilder.add(0, 489, -4.88440907254626E-5)
      matrixBuilder.add(1, 489, 4.8844090725462606E-5)
      matrixBuilder.add(0, 490, -4.324392733454803E-5)
      matrixBuilder.add(1, 490, 4.324392733454811E-5)
      matrixBuilder.add(0, 496, 6.903351855620161E-5)
      matrixBuilder.add(1, 496, -6.90335185562012E-5)
      matrixBuilder.add(0, 511, 3.946505594172827E-4)
      matrixBuilder.add(1, 511, -3.946505594172831E-4)
      matrixBuilder.add(0, 512, 2.621745995919226E-4)
      matrixBuilder.add(1, 512, -2.621745995919226E-4)
      matrixBuilder.add(0, 517, -4.459475951170906E-5)
      matrixBuilder.add(1, 517, 4.459475951170901E-5)
      matrixBuilder.add(0, 539, 2.5417562428184555E-4)
      matrixBuilder.add(1, 539, -2.5417562428184555E-4)
      matrixBuilder.add(0, 540, 5.271781246228031E-4)
      matrixBuilder.add(1, 540, -5.271781246228032E-4)
      matrixBuilder.add(0, 568, 1.860255150352447E-4)
      matrixBuilder.add(1, 568, -1.8602551503524485E-4)
      matrixBuilder.result.toDense.data
    }

    val classificationMultinomialIntercepts: Vector = {
      Vectors.dense(-0.12065879445860686, 0.12065879445860686)
    }

    val classificationModelSummary: Array[Double] = {
      Array(
        0.6833149135741672, 0.6662875751473734, 0.6217068546034618, 0.6127265245887887, 0.6060347986802873, 0.6031750687571562, 0.5969621534836274, 0.5940743031983118,
        0.5906089243339022, 0.5894724576491042, 0.5882187775729587)
    }

    val classificationROCArea: Double = 1.0

    val classificationROC: Array[(Double, Double)] = Array((0.0, 0.0), (0.0, 0.017543859649122806), (0.0, 0.03508771929824561), (0.0, 0.05263157894736842), (0.0, 0.07017543859649122))

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
