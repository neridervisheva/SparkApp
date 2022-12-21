import org.apache.spark.sql.{SparkSession, Row}
import org.apache.spark.sql.functions._

import org.apache.spark.sql.types.{StructType, StringType, IntegerType};

import org.apache.log4j.{Level, Logger}

object App {

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.WARN)

    val spark = SparkSession.builder().config("spark.master", "local")
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    import spark.implicits._

    // LOADING THE DATA

    // Create a schema for the DataFrame
    val schema = new StructType()
        .add("Year",IntegerType,true)
        .add("Month",IntegerType,true)
        .add("DayofMonth",IntegerType,true)
        .add("DayOfWeek",IntegerType,true)
        .add("DepTime",IntegerType,true)
        .add("CRSDepTime",IntegerType,true)
        .add("ArrTime",IntegerType,true)
        .add("CRSArrTime",IntegerType,true)
        .add("UniqueCarrier",StringType,true)
        .add("FlightNum",IntegerType,true)
        .add("TailNum",StringType,true)
        .add("ActualElapsedTime",IntegerType,true)
        .add("CRSElapsedTime",IntegerType,true)
        .add("AirTime",IntegerType,true)
        .add("ArrDelay",IntegerType,true)
        .add("DepDelay",IntegerType,true)
        .add("Origin",StringType,true)
        .add("Dest",StringType,true)
        .add("Distance",IntegerType,true)
        .add("TaxiIn",IntegerType,true)
        .add("TaxiOut",IntegerType,true)
        .add("Cancelled",IntegerType,true)
        .add("CancellationCode",StringType,true)
        .add("Diverted",IntegerType,true)
        .add("CarrierDelay",IntegerType,true)
        .add("WeatherDelay",IntegerType,true)
        .add("NASDelay",IntegerType,true)
        .add("SecurityDelay",IntegerType,true)
        .add("LateAircraftDelay",IntegerType,true)

    // Load data into DataFrame
    val df = spark.read.format("csv")
        .option("header", "true")
        .schema(schema)
        .load("src/main/resources/2008.csv.bz2")

    // Remove forbidden variables
    val df1 = df.drop("ArrTime").drop("ActualElapsedTime").drop("AirTime")
        .drop("TaxiIn").drop("Diverted").drop("CarrierDelay")
        .drop("WeatherDelay").drop("NASDelay").drop("SecurityDelay")
        .drop("LateAircraftDelay")

    df1.printSchema()
    df1.show(5, truncate = false)

    println("Number of instances: " + df1.count)

    // PROCESSING THE DATA

    // Remove instances of cancelled flights
    val df2 = df1.filter($"Cancelled".equalTo("0"))

    println("Number of instances: " + df2.count)

    println("Missing values for each variable:")
    df1.select(df2.columns
        .map(c => sum(col(c).isNull.cast("int")).alias(c)): _*).show

    // There are variables with missing values: DeptTime, TailNum,
    // CRSElapsedTime, ArrDelay, TaxiOut and CancellationCode

    // CancellationCode has more than 97% missing -> remove it
    val df3 = df2.drop("CancellationCode")

    println("Unique values of each variable:")
    df3.select("Year").distinct().show()
    df3.select("Month").distinct().show()
    df3.select("DayofMonth").distinct().show()
    df3.select("DayOfWeek").distinct().show()
    df3.select("DepTime").distinct().show()
    df3.select("CRSDepTime").distinct().show()
    df3.select("CRSArrTime").distinct().show()
    df3.select("UniqueCarrier").distinct().show()
    df3.select("FlightNum").distinct().show()
    df3.select("TailNum").distinct().show()
    df3.select("CRSElapsedTime").distinct().show()
    df3.select("ArrDelay").distinct().show()
    df3.select("DepDelay").distinct().show()
    df3.select("Origin").distinct().show()
    df3.select("Dest").distinct().show()
    df3.select("Distance").distinct().show()
    df3.select("TaxiOut").distinct().show()
    df3.select("Cancelled").distinct().show()

    // Month: 1, 2, 3, 4, ... to January, February, March, April...?
    // DayOfWeek: 1, 2, ... to Monday, Tuesday, ...?
    // Que es TaxiOut?

    // CRSElapsedTime = CRSArrTime - CRSDepTime
    // Hay algunos valores que están mal en CRSElapsedTime, corregirlos?
    // Hay que pensar mejor la formula para restar, ya que no se cumple siempre
    // Convertirlo a minutos o dejarlo en formato hhmm?
    //var CRSElapsedTime_new = df3.select("CRSArrTime", "CRSDepTime", "CRSElapsedTime")
    //    .withColumn("CRSElapsedTimeNew", col("CRSArrTime") - col("CRSDepTime"))
    //CRSElapsedTime_new.show()

    // DeptTime = CRSDepTime + DepDelay
    // Hay que convertir DepDelay de minutos a hhmm



  }
}
