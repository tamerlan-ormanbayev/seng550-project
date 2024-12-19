#Importing PySpark Libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, when, mean
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
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
df.printSchema()

#Preprocessing the data (Replacing/Handling missing values)
#Finding the mean for the CO levels in both sites
mean_CO_Winetavern = df.select(mean('CO_Winetavern')).first()[0]
mean_CO_Coleraine = df.select(mean('CO_Coleraine')).first()[0]

df = df.na.fill({'CO_Winetavern': mean_CO_Winetavern, 'CO_Coleraine': mean_CO_Coleraine})

#Drop rows where all columns are empty, just in case.
df = df.na.fill(0)  # Fill all null values with 0


#Building the Random Forest Classifier Model using only numerical features
assembler = VectorAssembler(inputCols=['CO_Winetavern', '8hr_Winetavern', 'Flag_Winetavern', 'CO_Coleraine', '8hr_Coleraine', 'Flag_Coleraine'], outputCol="features")
df = assembler.transform(df)

df = df.withColumn("exceeds_threshold", when((col("8hr_Winetavern") > 10) | (col("8hr_Coleraine") > 10), 1).otherwise(0))
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

rfc = RandomForestClassifier(featuresCol="features", labelCol="exceeds_threshold", numTrees=100)
rfc_model = rfc.fit(train_data)

#Perform Predictions on the test data
predictions = rfc_model.transform(test_data)

#Evaluate the model using the test data
auc_eval = BinaryClassificationEvaluator(labelCol="exceeds_threshold", rawPredictionCol="prediction", metricName="areaUnderROC")
a_eval = MulticlassClassificationEvaluator(labelCol="exceeds_threshold", predictionCol="prediction", metricName="accuracy")
p_eval= MulticlassClassificationEvaluator(labelCol="exceeds_threshold", predictionCol="prediction", metricName="weightedPrecision")
r_eval= MulticlassClassificationEvaluator(labelCol="exceeds_threshold", predictionCol="prediction", metricName="weightedRecall")
f1_eval = MulticlassClassificationEvaluator(labelCol="exceeds_threshold", predictionCol="prediction", metricName="f1")

auc_roc = auc_eval.evaluate(predictions)
accuracy = a_eval.evaluate(predictions)
precision = p_eval.evaluate(predictions)
recall = r_eval.evaluate(predictions)
f1_score = f1_eval.evaluate(predictions)

#Display the evaluation metrics
print("Area Under ROC: ", auc_roc)
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1_score)


#Finding Training Error
train_predictions = rfc_model.transform(train_data)
train_error = 1.0 - a_eval.evaluate(train_predictions)
print("Training Error: ", train_error)



