#!/usr/bin/env python
# coding: utf-8

# # Import libraries

import sys, os

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


# **Arguments from command line**
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

# **Remove forbidden variables**

# In[9]:





# In[11]:


df.show(5, False)


# # Analysis

# In[12]:


(df.count(), len(df.columns))


# ## Univariate analysis

# In[13]:


df.columns


# In[14]:


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



df = df.distinct()


# In[13]:


df.groupBy(df.columns).count().filter("count > 1").show()


# **Remove instances of cancelled flights**

# In[14]:


df.groupBy('Cancelled').count().show()


# In[15]:


df.groupBy('CancellationCode').count().show()


# In[16]:


df = df.filter(df.Cancelled == 0)


# In[17]:


df = df.filter(df.CancellationCode.isNull())


# **CancellationCode has more than 97% missing so it is removed**

# In[18]:


df = df.drop('CancellationCode', 'Cancelled')


# **Analyze missing values**

# In[19]:


size=df.count()


# In[21]:


print(tabulate([[c, df.filter(col(c).isNull()).count()/size] for c in df.columns], headers=['Name', 'Count %']))


# **TailNum and CRSElapsedTime can not be imputed from any other column, and ArrDelay is the target variable, but their number of missing values is not significant taking into account the total number, so the rows containing those missing values are removed**

# In[20]:


df = df.na.drop()


# In[21]:


(df.count(), len(df.columns))


# ## !!!!!Date preprocess

# In[22]:


df = df.withColumn(
    "Season", when((df.Month>2) & (df.Month<6), 1).when((df.Month>5) & (df.Month<9), 2
        ).when((df.Month>8) & (df.Month<12), 3).otherwise(4)
)
df.groupBy('Season').count().show()


# In[23]:


df = df.withColumn(
    "Date", F.date_format(F.expr("make_date(Year, Month, DayofMonth)"), "MM/dd/yyyy")
)

df.groupBy('Date').count().show()


# **Check that DepDelay = DepTime - CRSDepTime**

# In[24]:


df = df.withColumn("DepTimeNew", when(F.length(df.DepTime) == 3, concat(lit("0"),df.DepTime))         .when(F.length(df.DepTime) == 2, concat(lit("00"),df.DepTime))         .when(F.length(df.DepTime) == 1, concat(lit("000"),df.DepTime))         .otherwise(df.DepTime))


# In[25]:


df = df.withColumn("CRSDepTimeNew", when(F.length(df.CRSDepTime) == 3, concat(lit("0"),df.CRSDepTime))         .when(F.length(df.CRSDepTime) == 2, concat(lit("00"),df.CRSDepTime))         .when(F.length(df.CRSDepTime) == 1, concat(lit("000"),df.CRSDepTime))         .otherwise(df.CRSDepTime))


# In[26]:


df = df.withColumn("CRSArrTimeNew", when(F.length(df.CRSArrTime) == 3, concat(lit("0"),df.CRSArrTime))         .when(F.length(df.CRSArrTime) == 2, concat(lit("00"),df.CRSArrTime))         .when(F.length(df.CRSArrTime) == 1, concat(lit("000"),df.CRSArrTime))         .otherwise(df.CRSArrTime))


# In[27]:


df.select("DepTime","DepTimeNew","CRSDepTime","CRSDepTimeNew","CRSArrTime","CRSArrTimeNew").show()


# In[28]:


df.select("DepTimeNew","CRSDepTimeNew","DepDelay").show()


# In[29]:


df = df.withColumn("CRSDepTimeNewHour", substring(df.CRSDepTimeNew, 1,2))     .withColumn("CRSDepTimeNewMinute", substring(df.CRSDepTimeNew, 3,2))     .withColumn("DepTimeNewHour", substring(df.DepTimeNew, 1,2))     .withColumn("DepTimeNewMinute", substring(df.DepTimeNew, 3,2))

df = df.withColumn("CRSArrTimeNewHour", substring(df.CRSArrTimeNew, 1,2))     .withColumn("CRSArrTimeNewMinute", substring(df.CRSArrTimeNew, 3,2))


# In[30]:


df.select("CRSDepTimeNew","CRSDepTimeNewHour","CRSDepTimeNewMinute","DepTimeNew","DepTimeNewHour","DepTimeNewMinute").show()


# In[31]:


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

df.select("CRSDepTimeNew", "CRSArrTimeNew", "hourTimeCRSDep", "hourTimeCRSArr").show()


# ## Concordancy between related variables

# **No flights with same Origin and Destination**

# In[32]:


# quitarlos directamente para automatizar el proceso?
df.filter(df.Origin == df.Dest).show()


# In[33]:


df.columns


# ## Input

# ## Fix format of variables

# In[34]:


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


# In[35]:


df.columns


# ## Categoricals

# In[36]:


#convert the categorical attributes to binary features
categoricalAttributes = ['Origin', 'Dest', 'UniqueCarrier']

#Build a list of pipelist stages for the machine learning pipeline. 
#start by the feature transformer of one hot encoder for building the categorical features
pipelineStages = []
for columnName in categoricalAttributes:
    stringIndexer = StringIndexer(inputCol=columnName, outputCol=columnName+ "Index")
    pipelineStages.append(stringIndexer)
    oneHotEncoder = OneHotEncoder(inputCol=columnName+ "Index", outputCol=columnName + "Vec")
    pipelineStages.append(oneHotEncoder)
    
    
print("%s string indexer and one hot encoders transformers" %  len(pipelineStages) )


# In[37]:


# Combine all the feature columns into a single column in the dataframe
numericColumns = ['Year', 'Month', 'DayofMonth', 'DayOfWeek',
 'CRSElapsedTime', 'DepDelay', 'Distance', 'TaxiOut', 'Season',
 'hourTimeCRSDep', 'hourTimeCRSArr']
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


# In[38]:


scaled_df = featureOnlyModel.transform(df)


# In[39]:


# Inspect the result
scaled_df.select("features", "features_scaled").show(10, truncate=False)


# # Split

# In[42]:


# split into training(80%) and test(20%) datasets
df_train, df_test = scaled_df.randomSplit([.8,.2])

print("Num of training observations : %s" % df_train.count())
print("Num of test observations : %s" % df_test.count())


# In[46]:


df_train.select("features_scaled", "ArrDelay").show(2)


# # LinearRegression

# In[47]:


lr = LinearRegression(featuresCol = 'features_scaled', labelCol='ArrDelay', maxIter=10, regParam=0.3, elasticNetParam=0.8)


# In[48]:


lr_model = lr.fit(df_train)
print("Coefficients: " + str(lr_model.coefficients))
print("Intercept: " + str(lr_model.intercept))


# In[49]:


trainingSummary = lr_model.summary
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)


# In[ ]:


trainingSummary = lr_model.summary
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)


# **Evaluate**

# In[53]:


# Generate predictions
predictions = lr_model.transform(df_test)


# In[57]:


# Extract the predictions and the "known" correct labels
predandlabels = predictions.select("prediction", "ArrDelay")


# In[58]:


predandlabels.show()


# In[71]:


evaluatorRMSE = RegressionEvaluator(predictionCol="prediction", labelCol='ArrDelay', metricName='rmse')
evaluatorMAE = RegressionEvaluator(predictionCol="prediction", labelCol='ArrDelay', metricName='mae')
evaluatorR2 = RegressionEvaluator(predictionCol="prediction", labelCol='ArrDelay', metricName='r2')

print("RMSE: {0}".format(evaluatorRMSE.evaluate(predandlabels)))
print("MAE: {0}".format(evaluatorMAE.evaluate(predandlabels)))
print("R2: {0}".format(evaluatorR2.evaluate(predandlabels)))


# In[59]:


# Get the RMSE
print("RMSE: {0}".format(lr_model.summary.rootMeanSquaredError))
print("MAE: {0}".format(lr_model.summary.meanAbsoluteError))

# Get the R2
print("R2: {0}".format(lr_model.summary.r2))


# # GridSearch

# In[75]:


from pyspark.ml.classification import LogisticRegression

# Configure an machine learning pipeline, which consists of the 
# an estimator (classification) (Logistic regression)
lr = LinearRegression(featuresCol = 'features_scaled', 
                        labelCol='ArrDelay', maxIter=10, regParam=0.3, 
                        elasticNetParam=0.8)

# Fit the pipeline to create a model from the training data
lr_Model = lr.fit(df_train)

def getAccuracyForPipelineModel(featuresDf, model):
    #perform prediction using the featuresdf and pipelineModel
    #compute the accuracy in percentage float
    predictions = model.transform(featuresDf)
    predandlabels = predictions.select("prediction", "ArrDelay")
    return evaluatorRMSE.evaluate(predandlabels), evaluatorMAE.evaluate(predandlabels),             evaluatorR2.evaluate(predandlabels),  predictions.prediction

# Evaluating the model on training data
RMSETrainAccuracy, MAETrainAccuracy, R2TrainAccuracy, lrTrainResultDf = getAccuracyForPipelineModel(df_train, lr_Model)

# Repeat on test data
RMSETestAccuracy, MAETestAccuracy, R2TestAccuracy, lrTestResultDf = getAccuracyForPipelineModel(df_test, lr_Model)

print("==========================================")
print("LogisticRegression Model training accuracy (%) = " + str(R2TestAccuracy))
print("LogisticRegression Model test accuracy (%) = " + str(R2TrainAccuracy))
print("==========================================")


# **Hyperparameter Tuning with Grid search**

# In[76]:


maxIterRange = [5, 10, 30, 50, 100]
regParamRange = [1e-10, 1e-5, 1e-1]
#baseline values from previous section
bestIter = 10
bestRegParam = 0.01
bestModel = lr
bestAccuracy = R2TestAccuracy


# In[83]:



#for plotting purpose
iterations = []
regParams = []
accuracies = []
for maxIter in maxIterRange:
    for rp in regParamRange:
        currentLr = LinearRegression(featuresCol = 'features_scaled', 
                      labelCol='ArrDelay', maxIter=maxIter, regParam=rp)
        model = currentLr.fit(df_train)
        
        #use validation dataset test for accuracy
        RMSEAccuracy, MAEAccuracy, R2Accuracy, lrResultDf = getAccuracyForPipelineModel(df_train, model)
        
        accuracy=R2Accuracy
        print("maxIter: %s, regParam: %s, accuracy: %s " % (maxIter, rp, accuracy))
        accuracies.append(accuracy)
        regParams.append(rp)
        iterations.append(maxIter)
        
        if accuracy > bestAccuracy :
            bestIter = maxIter
            bestRegParam = rp
            bestModel = model
            bestAccuracy = accuracy


print("Best parameters: maxIter %s, regParam %s, accuracy : %s" % (bestIter, bestRegParam, bestAccuracy))

# Repeat on test data
RMSETestAccuracy, MAETestAccuracy, R2TestAccuracy, lrTestResultDf = getAccuracyForPipelineModel(df_test, lr_Model)

print("==========================================")
print("LinearRegression Model training accuracy (%) = " + str(R2TestAccuracy))
print("LinearRegression Model test accuracy (%) = " + str(R2TrainAccuracy))
print("==========================================")


# # CrossValidator

# In[91]:


# We use a ParamGridBuilder to construct a grid of parameters to search over.
grid = (ParamGridBuilder()
        .addGrid(lr.maxIter, maxIterRange) 
        .addGrid(lr.regParam,regParamRange )
        .build())

currentLr = LinearRegression(featuresCol = 'features_scaled', 
                      labelCol='ArrDelay')

crossValidator = CrossValidator(estimator=currentLr, 
                                estimatorParamMaps=grid, 
                                numFolds=5,
                                evaluator=evaluatorMAE)


# In[92]:


# Run cross-validation, and choose the best model
bestCvModel = crossValidator.fit(df_train)

# Evaluating the model on training data
RMSETrainAccuracy, MAETrainAccuracy, R2TrainAccuracy, lrTrainResultDf = getAccuracyForPipelineModel(df_train, bestCvModel)

# Repeat on test data
RMSETestAccuracy, MAETestAccuracy, R2TestAccuracy, lrTestResultDf = getAccuracyForPipelineModel(df_test, bestCvModel)

print("==========================================")
print("LogisticRegression Model training accuracy (%) = " + str(MAETestAccuracy))
print("LogisticRegression Model test accuracy (%) = " + str(MAETestAccuracy))
print("==========================================")


# In[ ]:


# Coefficients for the model
linearModel.coefficients


# In[ ]:


# Intercept for the model
linearModel.intercept


# In[ ]:


coeff_df = pd.DataFrame({"Feature": ["Intercept"] + featureCols, "Co-efficients": np.insert(linearModel.coefficients.toArray(), 0, linearModel.intercept)})
coeff_df = coeff_df[["Feature", "Co-efficients"]]
coeff_df

def getAccuracy(featuresDf, model):
    #perform prediction using the featuresdf and pipelineModel
    #compute the accuracy in percentage float
    predictions = model.transform(featuresDf)
    predandlabels = predictions.select("prediction", "ArrDelay")
    return evaluatorRMSE.evaluate(predandlabels), evaluatorMAE.evaluate(predandlabels), \
            evaluatorR2.evaluate(predandlabels),  predictions.prediction

def linearRegression(df_train, df_test, maxIter=10, regParam=0.3, elasticNetParam=0.8):
	lr = LinearRegression(featuresCol = 'features_scaled', labelCol='ArrDelay', maxIter=maxIter, regParam=regParam, elasticNetParam=elasticNetParam)
	lr_model = lr.fit(df_train)

	trainingSummary = lr_model.summary
	print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
	print("r2: %f" % trainingSummary.r2)

	# Generate predictions
	predictions = lr_model.transform(df_test)

	# Extract the predictions and the "known" correct labels
	predandlabels = predictions.select("prediction", "ArrDelay")


	evaluatorRMSE = RegressionEvaluator(predictionCol="prediction", labelCol='ArrDelay', metricName='rmse')
	evaluatorMAE = RegressionEvaluator(predictionCol="prediction", labelCol='ArrDelay', metricName='mae')
	evaluatorR2 = RegressionEvaluator(predictionCol="prediction", labelCol='ArrDelay', metricName='r2')

	print("RMSE: {0}".format(evaluatorRMSE.evaluate(predandlabels)))
	print("MAE: {0}".format(evaluatorMAE.evaluate(predandlabels)))
	print("R2: {0}".format(evaluatorR2.evaluate(predandlabels)))

def gridSearch(df_train, df_test, maxIter=10, regParam=0.3, elasticNetParam=0.8):

	maxIterRange = [5, 10, 30, 50, 100]
	regParamRange = [1e-10, 1e-5, 1e-1]
	#baseline values from previous section
	bestIter = -1
	bestRegParam = -1
	bestModel = lr
	bestAccuracy = 0



	#for plotting purpose
	iterations = []
	regParams = []
	accuracies = []
	for maxIter in maxIterRange:
	    for rp in regParamRange:
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
	        
	        if accuracy > bestAccuracy :
	            bestIter = maxIter
	            bestRegParam = rp
	            bestModel = model
	            bestAccuracy = accuracy


	print("Best parameters: maxIter %s, regParam %s, accuracy : %s" % (bestIter, bestRegParam, bestAccuracy))

	# Repeat on test data
	RMSETestAccuracy, MAETestAccuracy, R2TestAccuracy, lrTestResultDf = getAccuracy(df_test, lr_Model)


	evaluatorRMSE = RegressionEvaluator(predictionCol="prediction", labelCol='ArrDelay', metricName='rmse')
	evaluatorMAE = RegressionEvaluator(predictionCol="prediction", labelCol='ArrDelay', metricName='mae')
	evaluatorR2 = RegressionEvaluator(predictionCol="prediction", labelCol='ArrDelay', metricName='r2')


	print("==========================================")
	print("LinearRegression Model training accuracy (R2) = " + str(R2TestAccuracy))
	print("LinearRegression Model test accuracy (R2) = " + str(R2TrainAccuracy))
	print("==========================================")

def crossValidator(df_train, df_test, maxIter=10, regParam=0.3, elasticNetParam=0.8):


	evaluatorRMSE = RegressionEvaluator(predictionCol="prediction", labelCol='ArrDelay', metricName='rmse')
	evaluatorMAE = RegressionEvaluator(predictionCol="prediction", labelCol='ArrDelay', metricName='mae')
	evaluatorR2 = RegressionEvaluator(predictionCol="prediction", labelCol='ArrDelay', metricName='r2')


	# We use a ParamGridBuilder to construct a grid of parameters to search over.
	grid = (ParamGridBuilder()
	        .addGrid(lr.maxIter, maxIterRange) 
	        .addGrid(lr.regParam,regParamRange )
	        .build())

	currentLr = LinearRegression(featuresCol = 'features_scaled', 
	                      labelCol='ArrDelay')

	crossValidator = CrossValidator(estimator=currentLr, 
	                                estimatorParamMaps=grid, 
	                                numFolds=5,
	                                evaluator=evaluatorMAE)

	# Run cross-validation, and choose the best model
	bestCvModel = crossValidator.fit(df_train)

	# Evaluating the model on training data
	RMSETrainAccuracy, MAETrainAccuracy, R2TrainAccuracy, lrTrainResultDf = getAccuracyForPipelineModel(df_train, bestCvModel)

	# Repeat on test data
	RMSETestAccuracy, MAETestAccuracy, R2TestAccuracy, lrTestResultDf = getAccuracyForPipelineModel(df_test, bestCvModel)

	print("==========================================")
	print("LogisticRegression Model training accuracy (%) = " + str(MAETestAccuracy))
	print("LogisticRegression Model test accuracy (%) = " + str(MAETestAccuracy))
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

def loadData(files_path):
	df = spark.read.csv(path=files_path[0], schema=schema, header=True)

	for f in files_path[1:]:
	    df = df.union(spark.read.csv(path=f, schema=schema, header=True))

	print("Printing loaded dataframe schema", df.printSchema())

	return df


def main(dir_path, args = []):
	spark = SparkSession.builder.appName("BigDataApp-Group6").getOrCreate()
	

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

	df = loadData(files_path)

	#We remove forbidden variables
	df = removeForbidden(df)

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

	if '--path' in sys.argv:
		i = sys.argv.index('--path')
	    dir_path = sys.argv[i+1]

	    a.pop(i)
	    a.pop(i)

	    main(path, sys.argv)
	else:
	    print('Add the argument "--path" followed by the address where the data is located.')
	    exit(-1)
    # Read information about how to connect back to the JVM from the environment.
    
