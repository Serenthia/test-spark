name := "testSpark"

version := "1.0"

scalaVersion := "2.11.8"

libraryDependencies += "org.apache.spark" %% "spark-core"           % "2.2.0"
libraryDependencies += "org.apache.spark" %% "spark-mllib"          % "2.2.0"
libraryDependencies += "org.apache.spark" %% "spark-sql"            % "2.2.0"
libraryDependencies += "org.scalacheck"   %% "scalacheck"           % "1.13.5" % "test"
libraryDependencies += "org.specs2"       %% "specs2-core"          % "3.9.1"  % "test"
libraryDependencies += "org.specs2"       %% "specs2-mock"          % "3.9.1"  % "test"
libraryDependencies += "org.specs2"       %% "specs2-matcher-extra" % "3.9.1"  % "test"