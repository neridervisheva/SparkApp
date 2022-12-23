
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

    //df1.printSchema()
    //df1.show(5, truncate = false)

    // PROCESSING THE DATA

    // Remove duplicated rows
    val df1_distinct = df1.distinct()

    // Remove instances of cancelled flights
    val df2 = df1_distinct.filter($"Cancelled".equalTo("0"))

    //println("Number of instances: " + df2.count)

    println("Missing values for each variable:")
    //df1.select(df2.columns
    //    .map(c => sum(col(c).isNull.cast("int")).alias(c)): _*).show

    // There are variables with missing values: DeptTime, TailNum,
    // CRSElapsedTime, ArrDelay, TaxiOut and CancellationCode

    // CancellationCode has more than 97% missing -> remove it
    var df3 = df2.drop("CancellationCode")

    // CRSElapsedTime = CRSArrTime - CRSDepTime
    // There are some wrong values for CRSElapsedTime, so all of the rows are imputed
    // Convertirlo a hhmm o dejarlo en minutos?

    // CORRECCION: This can not be imputed this way because as the ArrTime and DeptTime
    // are in local time, the time zone could be different, not important because only
    // the values for a few rows are missing

    // Put time in correct format, full 4 digits
    //val df4 = df3.withColumn("CRSArrTimeNew", when(length(col("CRSArrTime")) === 3, concat(lit("0"),col("CRSArrTime")))
    //    .when(length(col("CRSArrTime")) === 2, concat(lit("00"),col("CRSArrTime")))
    //    .otherwise(col("CRSArrTime")))

    //var df5 = df4.withColumn("CRSDepTimeNew", when(length(col("CRSDepTime")) === 3, concat(lit("0"),col("CRSDepTime")))
    //    .when(length(col("CRSDepTime")) === 2, concat(lit("00"),col("CRSDepTime")))
    //    .otherwise(col("CRSDepTime")))

    // New auxiliar columns for calculations
    //df5 = df5.withColumn("CRSArrTimeNewHour", substring(col("CRSArrTimeNew"), 1,2))
    //df5 = df5.withColumn("CRSArrTimeNewMinute", substring(col("CRSArrTimeNew"), 3,2))

    //df5 = df5.withColumn("CRSDepTimeNewHour", substring(col("CRSDepTimeNew"), 1,2))
    //df5 = df5.withColumn("CRSDepTimeNewMinute", substring(col("CRSDepTimeNew"), 3,2))

    //df5.select("CRSArrTime","CRSArrTimeNew","CRSArrTimeNewHour","CRSArrTimeNewMinute").show(false)
    //df5.select("CRSDepTime","CRSDepTimeNew","CRSDepTimeNewHour","CRSDepTimeNewMinute").show(false)

    // Compute the real elapsed time
    //df5 = df5.withColumn("CRSElapsedTimeNew", when(col("CRSArrTimeNewHour") < col("CRSDepTimeNewHour"), (lit(24) + col("CRSArrTimeNewHour") - col("CRSDepTimeNewHour"))*60 - col("CRSDepTimeNewMinute") + col("CRSArrTimeNewMinute"))
    //    .otherwise((col("CRSArrTimeNewHour") - col("CRSDepTimeNewHour"))*60 - col("CRSDepTimeNewMinute") + col("CRSArrTimeNewMinute")))

    //df5 = df5.withColumn("CRSElapsedTimeNew",col("CRSElapsedTimeNew").cast(IntegerType))

    //df5.select("CRSArrTime","CRSDepTime","CRSElapsedTime","CRSElapsedTimeNew").show(false)

    // DepTime = CRSDepTime + DepDelay

    // Put time in correct format, full 4 digits
    df3 = df3.withColumn("CRSDepTimeNew", when(length(col("CRSDepTime")) === 3, concat(lit("0"),col("CRSDepTime")))
        .when(length(col("CRSDepTime")) === 2, concat(lit("00"),col("CRSDepTime")))
        .otherwise(col("CRSDepTime")))

    // New auxiliar columns for calculations
    df3 = df3.withColumn("CRSDepTimeNewHour", substring(col("CRSDepTimeNew"), 1,2))
    df3 = df3.withColumn("CRSDepTimeNewHourMinute", substring(col("CRSDepTimeNew"), 3,2))

    //df3 = df3.withColumn("DepDelayNew", when(col("DepDelay") === 0, col("DepDelayNew") = col("CRSDepTimeNew"))
    //    .when(col("DepDelay") < 60 && col("DepDelay") > 0, )
    //    .otherwise())


    df3.select("DepDelay","DepDelayNew").show(false)

    // Delete auxiliary columns

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

  }
}
