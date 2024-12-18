#Importing PySpark Libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, concat, when, split, mean
from pyspark.ml.evaluation import BinaryClassificationEvaluator 
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

# Verify the schema has the correct data types
df.printSchema()

#Handles Invalid values such as #DIV/0! by replacing them with None      
df = df.withColumn("CO_Winetavern", when(col("CO_Winetavern") == "#DIV/0!", lit(None)).otherwise(col("CO_Winetavern"))) \
       .withColumn("CO_Coleraine", when(col("CO_Coleraine") == "#DIV/0!", lit(None)).otherwise(col("CO_Coleraine"))) \
       .withColumn("8hr_Winetavern", when(col("8hr_Winetavern") == "#DIV/0!", lit(None)).otherwise(col("8hr_Winetavern"))) \
       .withColumn("8hr_Coleraine", when(col("8hr_Coleraine") == "#DIV/0!", lit(None)).otherwise(col("8hr_Coleraine")))

#Preprocessing the data (Replacing/Handling missing values)
#Finding the mean for the CO levels in both sites
mean_CO_Winetavern = df.select(mean('CO_Winetavern')).first()[0]
mean_CO_Coleraine = df.select(mean('CO_Coleraine')).first()[0]

df = df.na.fill({'CO_Winetavern': mean_CO_Winetavern, 'CO_Coleraine': mean_CO_Coleraine})

#Drop rows where all columns are empty, just in case.
df = df.dropna(how="all")

#To Test to see if Data is Cleaned
df.show(10)