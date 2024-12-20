#Importing Pandas
import pandas as pd

#Importing PySpark Libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, when, mean, regexp_extract, regexp_replace, trim, to_date, concat
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler

#Create a SparkSession
spark = SparkSession.builder.appName('CO_Data').config('spark.sql.debug.maxToStringFields', 1000).config('spark.driver.maxResultSize', '4g').config('spark.executor.memory', '8g').config('spark.sql.broadcastTimeout', '600').config('spark.sql.shuffle.partitions', '200').config('spark.sql.autoBroadcastJoinThreshold', -1).config('spark.executor.cores', '2').getOrCreate() 

#Reading the dataset
raw_data = spark.read.text('dublin-city-council-co-2011p20110929-1048.csv')
no_header = raw_data.rdd.zipWithIndex().filter(lambda x: 15 <= x[1] <= 3646).map(lambda x: x[0])

#Naming the columns so it is clear that a set of features belong to Winetavern Street Site and another set belong to Coleraine Street
cols = ['Date', 'Time', 'CO_Winetavern', '8hr_Winetavern', 'Flag_Winetavern', 'Comment_Winetavern', 'CO_Coleraine', '8hr_Coleraine', 'Flag_Coleraine', 'Comment_Coleraine']

#Creating a DataFrame from the raw data
df = spark.read.csv(no_header, header=False, inferSchema=True)
df = df.toDF(*cols)

#Data Clean Up - Parsing the CSV file's data correctly
df = df.withColumn("Date", regexp_extract(col("Date"), r"(\d{2}/\d{2}/\d{4})", 1)) \
       .withColumn("Date", to_date(col("Date"), "dd/MM/yyyy")) \
       .withColumn("Comment_Winetavern", trim(regexp_replace(col("Comment_Winetavern"), r"[')]", ""))) \
       .withColumn("Comment_Winetavern", when(col("Comment_Winetavern").isNull(), "").otherwise(col("Comment_Winetavern"))) \
       .withColumn("Comment_Coleraine", trim(regexp_replace(col("Comment_Coleraine"), r"[')]", ""))) \
       .withColumn("Comment_Coleraine", when(col("Comment_Coleraine").isNull(), "").otherwise(col("Comment_Coleraine")))

# Extracting Day, Month, and Year from Date for feature inclusion
df = df.withColumn("Day", col("Date").cast("date").cast("string").substr(9, 2).cast("int")) \
       .withColumn("Month", col("Date").cast("date").cast("string").substr(6, 2).cast("int")) \
       .withColumn("Year", col("Date").cast("date").cast("string").substr(1, 4).cast("int"))

#Handles Invalid values such as #DIV/0! by replacing them with None      
for col_name in ["CO_Winetavern", "CO_Coleraine", "8hr_Winetavern", "8hr_Coleraine"]:
    df = df.withColumn(col_name, when(col(col_name) == "#DIV/0!", lit(None)).otherwise(col(col_name).cast('double')))


#Changing the Flag Column Type into an Int Type
df = df.withColumn("Flag_Winetavern", col("Flag_Winetavern").cast('int')) \
       .withColumn("Flag_Coleraine", col("Flag_Coleraine").cast('int'))

#Preprocessing the data (Replacing/Handling missing values)
#Finding the mean for the CO levels and 8-hour rolling average columns in both sites
mean_CO_Winetavern = df.select(mean('CO_Winetavern')).first()[0]
mean_CO_Coleraine = df.select(mean('CO_Coleraine')).first()[0]

mean_8hr_Winetavern = df.select(mean('8hr_Winetavern')).first()[0]
mean_8hr_Coleraine = df.select(mean('8hr_Coleraine')).first()[0]

df = df.fillna({
    '8hr_Winetavern': mean_8hr_Winetavern,
    'Flag_Winetavern': 0,
    '8hr_Coleraine': mean_8hr_Coleraine,
    'Flag_Coleraine': 0
})

df = df.na.fill({'CO_Winetavern': mean_CO_Winetavern, 'CO_Coleraine': mean_CO_Coleraine})

#Adding target columns for each site based on the 8hr rolling average range
df = df.withColumn(
    "classification_winetavern",
    when(col("8hr_Winetavern") <= 0.2, 0)  #0 represents Low 
    .when(col("8hr_Winetavern") <= 0.4, 1) #1 represents Moderate
    .when(col("8hr_Winetavern") <= 0.6, 2) #2 represents High
    .otherwise(3)  #3 represents High
)

df = df.withColumn(
    "classification_coleraine",
    when(col("8hr_Coleraine") <= 0.2, 0)  #0 represents Low 
    .when(col("8hr_Coleraine") <= 0.4, 1) #1 represents Moderate
    .when(col("8hr_Coleraine") <= 0.6, 2) #2 represents High
    .otherwise(3)  #3 represents High
)

# Ensure classification columns are numeric
df = df.withColumn("classification_winetavern", col("classification_winetavern").cast("integer"))
df = df.withColumn("classification_coleraine", col("classification_coleraine").cast("integer"))

#Building Two Random Forest Classifiers for Winetavern Street and Coleraine Street respectively
assembler = VectorAssembler(inputCols=[
    'Time','CO_Winetavern', 'Flag_Winetavern', 'CO_Coleraine', 'Flag_Coleraine', 'Day', 'Month', 'Year'
], outputCol="features")

df = assembler.transform(df)

# Validate classifications and ensure they are non-null
df = df.fillna({'classification_winetavern': 0, 'classification_coleraine': 0})  # Fill missing classification values with 0

# Combine classifications into a single column
df = df.withColumn(
    "combined_classification",
    concat(col("classification_winetavern").cast("string"), lit("_"), col("classification_coleraine").cast("string"))
)

# Create fractions map for combined classifications
fractions = df.groupBy("combined_classification") \
    .count() \
    .rdd \
    .map(lambda row: (row["combined_classification"], row["count"])) \
    .collectAsMap()

# Handle empty fractions
fractions = {k: v for k, v in fractions.items() if k is not None}

# Normalize fractions for stratified sampling
total = sum(fractions.values())
fractions = {k: 0.8 * v / total for k, v in fractions.items()}

# Apply stratified sampling
train_data = df.sampleBy("combined_classification", fractions=fractions, seed=42)
test_data = df.subtract(train_data)

rfc_w = RandomForestClassifier(featuresCol="features", labelCol="classification_winetavern")
rfc_c = RandomForestClassifier(featuresCol="features", labelCol="classification_coleraine")

#Training WineTavern and Coleraine Classifiers with Cross-validation
paramGrid_w = ParamGridBuilder().addGrid(rfc_w.numTrees, [50, 100]).addGrid(rfc_w.maxDepth, [5, 10]).build()
crossval_w = CrossValidator(estimator=rfc_w, estimatorParamMaps=paramGrid_w, evaluator=MulticlassClassificationEvaluator(labelCol="classification_winetavern"), numFolds=2, parallelism=4)

cv_model_w = crossval_w.fit(train_data)
rfc_model_w = cv_model_w.bestModel        # Best Performing Winetavern Model

paramGrid_c = ParamGridBuilder().addGrid(rfc_c.numTrees, [50, 100]).addGrid(rfc_c.maxDepth, [5, 10]).build()
crossval_c = CrossValidator(estimator=rfc_c, estimatorParamMaps=paramGrid_c, evaluator=MulticlassClassificationEvaluator(labelCol="classification_coleraine"), numFolds=2, parallelism=4)

cv_model_c = crossval_c.fit(train_data)
rfc_model_c = cv_model_c.bestModel        # Best Performing Coleraine Model

#Make Predictions on each site separately
predictions_w = rfc_model_w.transform(test_data)
predictions_c = rfc_model_c.transform(test_data)

#Create Evalutor Model to Measure the model's ability to predict
eval = MulticlassClassificationEvaluator(predictionCol="prediction")

#Evaluate both models using the test data
acc_wine = eval.setMetricName("accuracy").setLabelCol("classification_winetavern").evaluate(predictions_w)
prec_wine = eval.setMetricName("weightedPrecision").setLabelCol("classification_winetavern").evaluate(predictions_w)
recall_wine = eval.setMetricName("weightedRecall").setLabelCol("classification_winetavern").evaluate(predictions_w)
f1_wine = eval.setMetricName("f1").setLabelCol("classification_winetavern").evaluate(predictions_w)

acc_col = eval.setMetricName("accuracy").setLabelCol("classification_coleraine").evaluate(predictions_c)
prec_col = eval.setMetricName("weightedPrecision").setLabelCol("classification_coleraine").evaluate(predictions_c)
recall_col = eval.setMetricName("weightedRecall").setLabelCol("classification_coleraine").evaluate(predictions_c)
f1_col = eval.setMetricName("f1").setLabelCol("classification_coleraine").evaluate(predictions_c)

# Finding Training Accuracy and Training Error for Winetavern Street and Coleraine Street
train_predictions_w = rfc_model_w.transform(train_data)
train_accuracy_w = eval.setMetricName("accuracy").setLabelCol("classification_winetavern").evaluate(train_predictions_w)
train_error_w = 1 - train_accuracy_w

train_predictions_c = rfc_model_c.transform(train_data)
train_accuracy_c = eval.setMetricName("accuracy").setLabelCol("classification_coleraine").evaluate(train_predictions_c)
train_error_c = 1 - train_accuracy_c

# Display the evaluation metrics
print("\nEvaluation Metrics")
print("Winetavern Street - Accuracy: ", acc_wine)
print("Winetavern Street - Precision: ", prec_wine)
print("Winetavern Street - Recall: ", recall_wine)
print("Winetavern Street - F1 Score: ", f1_wine)

print("Winetavern Street - Training Accuracy: ", train_accuracy_w)
print("Winetavern Street - Training Error: ", train_error_w)
print()  # Just to add some space

print("Coleraine Street - Accuracy: ", acc_col)
print("Coleraine Street - Precision: ", prec_col)
print("Coleraine Street - Recall: ", recall_col)
print("Coleraine Street - F1 Score: ", f1_col)

print("Coleraine Street - Training Accuracy: ", train_accuracy_c)
print("Coleraine Street - Training Error: ", train_error_c)
print()  # Just to add some space

# Display Confusion Matrix
print("Confusion Matrix for Winetavern Street:")
predictions_w.groupBy("classification_winetavern", "prediction").count().show()

print("Confusion Matrix for Coleraine Street:")
predictions_c.groupBy("classification_coleraine", "prediction").count().show()

# Group by classifications and count occurrences
df.groupBy("classification_winetavern").count().show()
df.groupBy("classification_coleraine").count().show()


