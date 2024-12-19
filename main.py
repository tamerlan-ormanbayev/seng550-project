#Importing PySpark Libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, when, mean
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler

#Create a SparkSession
spark = SparkSession.builder.appName('CO_Data').getOrCreate()

#Reading the dataset
raw_data = spark.read.text('dublin-city-council-co-2011p20110929-1048.csv')
no_header = raw_data.rdd.zipWithIndex().filter(lambda x: x[1] >= 7).map(lambda x: x[0])

#Naming the columns so it is clear that a set of features belong to Winetavern Street Site and another set belong to Coleraine Street
cols = ['Date', 'Time', 'CO_Winetavern', '8hr_Winetavern', 'Flag_Winetavern', 'Comment_Winetavern', 'CO_Coleraine', '8hr_Coleraine', 'Flag_Coleraine', 'Comment_Coleraine']

#Creating a DataFrame from the raw data
df = spark.read.csv(no_header, header=False, inferSchema=True)
df = df.toDF(*cols)

#Handles Invalid values such as #DIV/0! by replacing them with None      
df = df.withColumn("CO_Winetavern", when(col("CO_Winetavern") == "#DIV/0!", lit(None)).otherwise(col("CO_Winetavern").cast('double'))) \
       .withColumn("CO_Coleraine", when(col("CO_Coleraine") == "#DIV/0!", lit(None)).otherwise(col("CO_Coleraine").cast('double'))) \
       .withColumn("8hr_Winetavern", when(col("8hr_Winetavern") == "#DIV/0!", lit(None)).otherwise(col("8hr_Winetavern").cast('double'))) \
       .withColumn("8hr_Coleraine", when(col("8hr_Coleraine") == "#DIV/0!", lit(None)).otherwise(col("8hr_Coleraine").cast('double')))

#Changing the Flag Column Type into an Int Type
df = df.withColumn("Flag_Winetavern", col("Flag_Winetavern").cast('int')) \
       .withColumn("Flag_Coleraine", col("Flag_Coleraine").cast('int'))

# Verify the schema has the correct data types
#df.printSchema()

#Preprocessing the data (Replacing/Handling missing values)
#Finding the mean for the CO levels in both sites
mean_CO_Winetavern = df.select(mean('CO_Winetavern')).first()[0]
mean_CO_Coleraine = df.select(mean('CO_Coleraine')).first()[0]

df = df.na.fill({'CO_Winetavern': mean_CO_Winetavern, 'CO_Coleraine': mean_CO_Coleraine})

#Drop rows where all columns are empty, just in case.
df = df.na.fill(0)  # Fill all null values with 0

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

#Building Two Random Forest Classifiers for Winetavern Street and Coleraine Street respectively
assembler = VectorAssembler(inputCols=['CO_Winetavern', '8hr_Winetavern', 'Flag_Winetavern', 'CO_Coleraine', '8hr_Coleraine', 'Flag_Coleraine'], outputCol="features")
df = assembler.transform(df)
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

rfc_w = RandomForestClassifier(featuresCol="features", labelCol="classification_winetavern", numTrees=100)
rfc_model_w = rfc_w.fit(train_data)

rfc_c = RandomForestClassifier(featuresCol="features", labelCol="classification_coleraine", numTrees=100)
rfc_model_c = rfc_c.fit(train_data)

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

#Finding Training Accuracy and Training Error for Winetavern Street and Coleraine Street
train_predictions_w = rfc_model_w.transform(train_data)
train_accuracy_w = eval.setMetricName("accuracy").setLabelCol("classification_winetavern").evaluate(train_predictions_w)
train_error_w = 1 - train_accuracy_w

train_predictions_c = rfc_model_c.transform(train_data)
train_accuracy_c = eval.setMetricName("accuracy").setLabelCol("classification_coleraine").evaluate(train_predictions_c)
train_error_c = 1 - train_accuracy_c

#Display the evaluation metrics
print("\nEvaluation Metrics")
print("Winetavern Street - Accuracy: ", acc_wine)
print("Winetavern Street - Precision: ", prec_wine)
print("Winetavern Street - Recall: ", recall_wine)
print("Winetavern Street - F1 Score: ", f1_wine)

print("Winetavern Street - Training Accuracy: ", train_accuracy_w)
print("Winetavern Street - Training Error: ", train_error_w)
print() #Just to add some space

print("Coleraine Street - Accuracy: ", acc_col)
print("Coleraine Street - Precision: ", prec_col)
print("Coleraine Street - Recall: ", recall_col)
print("Coleraine Street - F1 Score: ", f1_col)

print("Coleraine Street - Training Accuracy: ", train_accuracy_c)
print("Coleraine Street - Training Error: ", train_error_c)
print() #Just to add some space


#Display Confusion Matrix
print("Confusion Matrix for Winetavern Street:")
predictions_w.groupBy("classification_winetavern", "prediction").count().show()

print("Confusion Matrix for Coleraine Street:")
predictions_c.groupBy("classification_coleraine", "prediction").count().show()





