#!/usr/bin/env python
# coding: utf-8

#Import libraries

import sys, os
import seaborn as sns


from pyspark.sql import SparkSession
from pyspark.sql.types import StructType,StructField, StringType, IntegerType
from pyspark.sql.functions import unix_timestamp, to_timestamp, round, log, col, isnan, when, count, concat, lit, substring, udf, desc, hour
import pyspark.sql.functions as F

from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.mllib.evaluation import RegressionMetrics

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, CrossValidatorModel
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder 
from pyspark.ml.evaluation import RegressionEvaluator


schema = StructType([
	StructField("Year",IntegerType(),nullable=True),
	StructField("Month",IntegerType(),nullable=True),
	StructField("DayofMonth",IntegerType(),nullable=True),
	StructField("DayOfWeek",IntegerType(),nullable=True),
	StructField("DepTime",IntegerType(),nullable=True),
	StructField("CRSDepTime",IntegerType(),nullable=True),
	StructField("ArrTime",IntegerType(),nullable=True),
	StructField("CRSArrTime",IntegerType(),nullable=True),
	StructField("UniqueCarrier",StringType(),nullable=True),
	StructField("FlightNum",IntegerType(),nullable=True),
	StructField("TailNum",StringType(),nullable=True),
	StructField("ActualElapsedTime",IntegerType(),nullable=True),
	StructField("CRSElapsedTime",IntegerType(),nullable=True),
	StructField("AirTime",IntegerType(),nullable=True),
	StructField("ArrDelay",IntegerType(),nullable=True),
	StructField("DepDelay",IntegerType(),nullable=True),
	StructField("Origin",StringType(),nullable=True),
	StructField("Dest",StringType(),nullable=True),
	StructField("Distance",IntegerType(),nullable=True),
	StructField("TaxiIn",IntegerType(),nullable=True),
	StructField("TaxiOut",IntegerType(),nullable=True),
	StructField("Cancelled",IntegerType(),nullable=True),
	StructField("CancellationCode",StringType(),nullable=True),
	StructField("Diverted",IntegerType(),nullable=True),
	StructField("CarrierDelay",IntegerType(),nullable=True),
	StructField("WeatherDelay",IntegerType(),nullable=True),
	StructField("NASDelay",IntegerType(),nullable=True),
	StructField("SecurityDelay",IntegerType(),nullable=True),
	StructField("LateAircraftDelay",IntegerType(),nullable=True)
])

def analysis(df):

	print("Shape: ", (df.count(), len(df.columns)))

	numerics_col = [
		'Year',
		'Month',
		'DayofMonth',
		'DayOfWeek',
		'ArrDelay',
		'DepDelay',
		'Distance',
		'TaxiOut']

	categorical_col = [
		'UniqueCarrier',
		'FlightNum',
		'TailNum',
		'Origin',
		'Dest',
		'Cancelled',
		'CancellationCode']

	for c in categorical_col:
		print(df.groupBy(c).count().show())

	for c in numerics_col:
		print(df.select(c).describe().show())

def getAccuracy(featuresDf, model):
	#perform prediction and real values to
	#compute the accuracy in percentage float
	evaluatorRMSE = RegressionEvaluator(predictionCol="prediction", labelCol='ArrDelay', metricName='rmse')
	evaluatorMAE = RegressionEvaluator(predictionCol="prediction", labelCol='ArrDelay', metricName='mae')
	evaluatorR2 = RegressionEvaluator(predictionCol="prediction", labelCol='ArrDelay', metricName='r2')

	predictions = model.transform(featuresDf)
	predandlabels = predictions.select("prediction", "ArrDelay")
	return evaluatorRMSE.evaluate(predandlabels), evaluatorMAE.evaluate(predandlabels), \
			evaluatorR2.evaluate(predandlabels),  predictions.prediction

def linearRegression(df_train, df_test, maxIter=10, regParam=0.3, elasticNetParam=0.8):
	#Linear Regression Model define and fit
	lr = LinearRegression(featuresCol = 'features_scaled', labelCol='ArrDelay', maxIter=maxIter, regParam=regParam, elasticNetParam=elasticNetParam)
	lr_model = lr.fit(df_train)

	trainingSummary = lr_model.summary
	print("Summary RMSE: %f" % trainingSummary.rootMeanSquaredError)
	print("Summary R2: %f" % trainingSummary.r2)

	# Generate predictions
	predictions = lr_model.transform(df_test)

	# Extract the predictions and the "known" correct labels
	predandlabels = predictions.select("prediction", "ArrDelay")

	RMSETrainAccuracy, MAETrainAccuracy, R2TrainAccuracy, lrTrainResultDf = getAccuracy(df_train, lr_model)

	# Repeat on test data
	RMSETestAccuracy, MAETestAccuracy, R2TestAccuracy, lrTestResultDf = getAccuracy(df_test, lr_model)

	print("==========================================")
	print("LinearRegression Model training accuracy (RMSET) = " + str(RMSETrainAccuracy))
	print("LinearRegression Model test accuracy (RMSET) = " + str(RMSETestAccuracy))
	print("==========================================")

def gridSearch(df_train, df_test, maxIter=10, regParam=0.3, elasticNetParam=0.8):

	maxIterRange = [5, 10, 30, 50, 100]
	regParamRange = [1e-10, 1e-5, 1e-1]
	#baseline values from previous section
	bestIter = -1
	bestRegParam = -1
	bestModel = LinearRegression(featuresCol = 'features_scaled', labelCol='ArrDelay')
	bestAccuracy = 0



	#for plotting purpose
	iterations = []
	regParams = []
	accuracies = []
	# we run through the possible combinations of the model's parameters
	for maxIter in maxIterRange:
		for rp in regParamRange:
			#Linear Regression Model define and fit
			currentLr = LinearRegression(featuresCol = 'features_scaled', 
						  labelCol='ArrDelay', maxIter=maxIter, regParam=rp)
			model = currentLr.fit(df_train)
			
			#use validation dataset test for accuracy
			RMSEAccuracy, MAEAccuracy, R2Accuracy, lrResultDf = getAccuracy(df_train, model)
			
			accuracy=R2Accuracy
			print("maxIter: %s, regParam: %s, accuracy: %s " % (maxIter, rp, accuracy))
			accuracies.append(accuracy)
			regParams.append(rp)
			iterations.append(maxIter)
			
			#check if this model is better than the previous ones
			if accuracy > bestAccuracy :
				bestIter = maxIter
				bestRegParam = rp
				bestModel = model
				bestAccuracy = accuracy


	print("Best parameters: maxIter %s, regParam %s, accuracy : %s" % (bestIter, bestRegParam, bestAccuracy))

	RMSETrainAccuracy, MAETrainAccuracy, R2TrainAccuracy, lrTrainResultDf = getAccuracy(df_train, bestModel)

	# Repeat on test data
	RMSETestAccuracy, MAETestAccuracy, R2TestAccuracy, lrTestResultDf = getAccuracy(df_test, bestModel)

	print("==========================================")
	print("LinearRegression Model training accuracy (RMSE) = " + str(RMSETrainAccuracy))
	print("LinearRegression Model test accuracy (RMSE) = " + str(RMSETestAccuracy))
	print("==========================================")

def crossValidator(df_train, df_test, maxIter=10, regParam=0.3, elasticNetParam=0.8):

	lr = LinearRegression(featuresCol = 'features_scaled', 
                        labelCol='ArrDelay', elasticNetParam=0.8)

	maxIterRange = [5, 10, 30, 50, 100]
	regParamRange = [1e-10, 1e-5, 1e-1]
	#baseline values from previous section
	bestIter = -1
	bestRegParam = -1
	bestModel = lr
	bestAccuracy = 0

	evaluatorRMSE = RegressionEvaluator(predictionCol="prediction", labelCol='ArrDelay', metricName='rmse')
	evaluatorMAE = RegressionEvaluator(predictionCol="prediction", labelCol='ArrDelay', metricName='mae')
	evaluatorR2 = RegressionEvaluator(predictionCol="prediction", labelCol='ArrDelay', metricName='r2')

	
	# We use a ParamGridBuilder to construct a grid of parameters to search over.
	grid = (ParamGridBuilder()
			.addGrid(lr.maxIter, maxIterRange) 
			.addGrid(lr.regParam,regParamRange )
			.build())

	crossValidator = CrossValidator(estimator=lr, 
									estimatorParamMaps=grid, 
									numFolds=5,
									evaluator=evaluatorMAE)

	# Run cross-validation, and choose the best model
	bestCvModel = crossValidator.fit(df_train)

	# Evaluating the model on training data
	RMSETrainAccuracy, MAETrainAccuracy, R2TrainAccuracy, lrTrainResultDf = getAccuracy(df_train, bestCvModel)

	# Repeat on test data
	RMSETestAccuracy, MAETestAccuracy, R2TestAccuracy, lrTestResultDf = getAccuracy(df_test, bestCvModel)

	print("==========================================")
	print("LogisticRegression Model training accuracy (RMSE) = " + str(RMSETrainAccuracy))
	print("LogisticRegression Model test accuracy (RMSE) = " + str(RMSETestAccuracy))
	print("==========================================")



def encoding(df, categoricalAttributes = ['Origin', 'Dest', 'UniqueCarrier'], numericColumns = ['Year', 'Month', 'DayofMonth', 'DayOfWeek',
 'CRSElapsedTime', 'DepDelay', 'Distance', 'TaxiOut', 'Season',  'hourTimeCRSDep', 'hourTimeCRSArr']):
	#convert the categorical attributes to binary features
	#Build a list of pipelist stages for the machine learning pipeline. 
	#start by the feature transformer of one hot encoder for building the categorical features
	pipelineStages = []
	for columnName in categoricalAttributes:
		stringIndexer = StringIndexer(inputCol=columnName, outputCol=columnName+ "Index")
		pipelineStages.append(stringIndexer)
		oneHotEncoder = OneHotEncoder(inputCol=columnName+ "Index", outputCol=columnName + "Vec")
		pipelineStages.append(oneHotEncoder)
		
		
	print("%s string indexer and one hot encoders transformers" %  len(pipelineStages) )


	# Combine all the feature columns into a single column in the dataframe
	categoricalCols = [s + "Vec" for s in categoricalAttributes]
	allFeatureCols =  numericColumns + categoricalCols
	vectorAssembler = VectorAssembler(
		inputCols=allFeatureCols,
		outputCol="features")
	pipelineStages.append(vectorAssembler)

	standardScaler = StandardScaler(inputCol="features", outputCol="features_scaled")
	pipelineStages.append(standardScaler)

	print("%s feature columns: %s" % (len(allFeatureCols),allFeatureCols))


	#Build pipeline for feature extraction
	featurePipeline = Pipeline(stages=pipelineStages)
	featureOnlyModel = featurePipeline.fit(df)


	scaled_df = featureOnlyModel.transform(df)

	return scaled_df

def processing(df):
	#Remove duplicated rows
	df = df.distinct()

	#We filter cancelled trips from some cause
	df = df.filter(df.Cancelled == 0)
	df = df.filter(df.CancellationCode.isNull())

	#We remove those columns
	df = df.drop('CancellationCode', 'Cancelled')
	df = df.na.drop()

	#We add some features such as the season or cheduled departure time (hour) 
	df = df.withColumn(
	"Season", when((df.Month>2) & (df.Month<6), 1).when((df.Month>5) & (df.Month<9), 2
		).when((df.Month>8) & (df.Month<12), 3).otherwise(4)
	)

	#To get the approximate time we adapt the time format. Then we stop the hours of the minutes and round to the nearest hour.
	df = df.withColumn("CRSDepTimeNew", when(F.length(df.CRSDepTime) == 3, concat(lit("0"),df.CRSDepTime)) \
		.when(F.length(df.CRSDepTime) == 2, concat(lit("00"),df.CRSDepTime)) \
		.when(F.length(df.CRSDepTime) == 1, concat(lit("000"),df.CRSDepTime)) \
		.otherwise(df.CRSDepTime))
	df = df.withColumn("CRSArrTimeNew", when(F.length(df.CRSArrTime) == 3, concat(lit("0"),df.CRSArrTime)) \
		.when(F.length(df.CRSArrTime) == 2, concat(lit("00"),df.CRSArrTime)) \
		.when(F.length(df.CRSArrTime) == 1, concat(lit("000"),df.CRSArrTime)) \
		.otherwise(df.CRSArrTime))

	df = df.withColumn("CRSDepTimeNewHour", substring(df.CRSDepTimeNew, 1,2)) \
	.withColumn("CRSDepTimeNewMinute", substring(df.CRSDepTimeNew, 3,2)) 

	df = df.withColumn("CRSArrTimeNewHour", substring(df.CRSArrTimeNew, 1,2)) \
		.withColumn("CRSArrTimeNewMinute", substring(df.CRSArrTimeNew, 3,2))

	df = df.withColumn(
		"datetime",
		F.date_format(
			F.expr("make_timestamp(Year, Month, DayofMonth, CRSDepTimeNewHour, CRSDepTimeNewMinute, 0)"),
			"dd/MM/yyyy HH:mm"
		)
	)
	df_truncated = hour((round(unix_timestamp(to_timestamp(col("datetime"),"dd/MM/yyyy HH:mm"))/3600)*3600).cast("timestamp"))
	df = df.withColumn("hourTimeCRSDep", df_truncated)

	df = df.withColumn(
		"datetime",
		F.date_format(
			F.expr("make_timestamp(Year, Month, DayofMonth, CRSArrTimeNewHour, CRSArrTimeNewMinute, 0)"),
			"dd/MM/yyyy HH:mm"
		)
	)
	df_truncated = hour((round(unix_timestamp(to_timestamp(col("datetime"),"dd/MM/yyyy HH:mm"))/3600)*3600).cast("timestamp"))
	df = df.withColumn("hourTimeCRSArr", df_truncated)


	#we check that there are no flights with origin and destination in the same place.
	df.filter(df.Origin == df.Dest).show()

	#We eliminate the generated temporary columns and others that we are not going to use.
	df = df.drop(
		'DepTime',
		'CRSDepTime',
		'CRSArrTime',
		'Date', 'datetime',
		'DepTimeNew',
		'CRSDepTimeNew',
		'CRSArrTimeNew',
		'CRSDepTimeNewHour',
		'CRSDepTimeNewMinute',
		'DepTimeNewHour',
		'DepTimeNewMinute',
		'CRSArrTimeNewHour',
		'CRSArrTimeNewMinute',
		'FlightNum', 'TailNum'
	)

	return df

def removeForbidden(df):
	df = df.drop("ArrTime").drop("ActualElapsedTime"
		).drop("AirTime").drop("TaxiIn").drop("Diverted"
		).drop("CarrierDelay").drop("WeatherDelay").drop("NASDelay"
		).drop("SecurityDelay").drop("LateAircraftDelay")

	return df

def loadData(spark, files_path):
	df = spark.read.csv(path=files_path[0], schema=schema, header=True)

	for f in files_path[1:]:
		df = df.union(spark.read.csv(path=f, schema=schema, header=True))

	print("Printing loaded dataframe schema", df.printSchema())

	return df


def main(dir_path, args = []):
	spark = SparkSession.builder.appName("BigDataApp-Group6").getOrCreate()
	spark.sparkContext.setLogLevel("WARN")

	if dir_path[-1] != '/':
		dir_path+='/'

	files_path = []
	for path in os.listdir(dir_path):
		# check if current path is a file
		if os.path.isfile(os.path.join(dir_path, path)) and path.endswith(".csv.bz2"):
			files_path.append(dir_path+path)
			
	if len(files_path) == 0:
		print('No files with extension "csv.bz2", try another directory.')
		exit(-1)

	df = loadData(spark, files_path)

	#We remove forbidden variables
	df = removeForbidden(df)

	if '--analysis' in args:
		analysis(df)

	print("Processing...")
	df = processing(df)

	scaled_df = encoding(df)

	#Split into training(80%) and test(20%) datasets
	df_train, df_test = scaled_df.randomSplit([.8,.2])

	print("Num of training observations : %s" % df_train.count())
	print("Num of test observations : %s" % df_test.count())


	#Models
	if '--LinearRegression' in args:
		linearRegression(df_train, df_test)
	elif '--GridSearch' in args:
		gridSearch(df_train, df_test)
	else:
		crossValidator(df_train, df_test)

	exit(0)




if __name__ == "__main__":

	print("\nArguments passed:", end = " ")
	for i in range(1, len(sys.argv)):
		print(sys.argv[i], end = " ")

	#Arguments from command line, we extract path from arguments
	if '--path' in sys.argv:
		i = sys.argv.index('--path')
		dir_path = sys.argv[i+1]

		sys.argv.pop(i)
		sys.argv.pop(i)

		main(dir_path, sys.argv)

		exit(0)
	else:
		print('Add the argument "--path" followed by the address where the data is located.')
		exit(-1)
	# Read information about how to connect back to the JVM from the environment.
	
