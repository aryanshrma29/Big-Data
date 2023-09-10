# -*- coding: utf-8 -*-


!pip install pyspark

import pyspark

from pyspark.sql import SparkSession

spark=SparkSession.builder.appName('big_data').getOrCreate()

spark

data = spark.read.csv('/content/train.csv')

data.show()
type(data)

data = spark.read.csv('/content/train.csv',header=True, inferSchema=True)

data.show()

from pyspark.sql.types import IntegerType, FloatType
from pyspark.sql.functions import col
data = data.withColumn("Age", col("Age").cast(IntegerType()))
data = data.withColumn("Num_of_Loan", col("Num_of_Loan").cast(IntegerType()))
data = data.withColumn("Num_of_Delayed_Payment", col("Num_of_Delayed_Payment").cast(IntegerType()))
data = data.withColumn("Changed_Credit_Limit", col("Changed_Credit_Limit").cast(FloatType()))
data = data.withColumn("Outstanding_Debt", col("Outstanding_Debt").cast(FloatType()))
data = data.withColumn("Amount_invested_monthly", col("Amount_invested_monthly").cast(FloatType()))
data = data.withColumn("Monthly_Balance", col("Monthly_Balance").cast(FloatType()))

data.printSchema()

data = data.drop('ID','Customer_ID','SSN','Month','Name')

data.show()

#cleaning the data
from pyspark.sql.functions import when, col
data = data.withColumn("Occupation", when(data.Occupation == "_______", None).otherwise(data.Occupation))
data = data.withColumn("Credit_Mix", when(data.Credit_Mix == "_", None).otherwise(data.Credit_Mix))
data = data.withColumn("Changed_Credit_Limit", when(data.Changed_Credit_Limit == "_", None).otherwise(data.Changed_Credit_Limit))
data = data.withColumn("Age", when(col('Age').rlike('_'), None).otherwise(data.Age))
data = data.withColumn("Age", when(col('Age').rlike('-'), None).otherwise(data.Age))
data = data.withColumn("Annual_Income", when(col('Annual_Income').rlike('_'), None).otherwise(data.Annual_Income))
data = data.withColumn("Num_of_Delayed_Payment", when(col('Num_of_Delayed_Payment').rlike('_'), None).otherwise(data.Num_of_Delayed_Payment))
data = data.withColumn("Num_of_Loan", when(col('Num_of_Loan').rlike('_'), None).otherwise(data.Num_of_Loan))
data = data.withColumn("Num_of_Loan", when(col('Num_of_Loan').rlike('-'), None).otherwise(data.Num_of_Loan))
data = data.withColumn("Amount_invested_monthly", when(col('Amount_invested_monthly').rlike('__'), None).otherwise(data.Amount_invested_monthly))
data = data.withColumn("Payment_Behaviour", when(col('Payment_Behaviour').rlike('!@9#%8'), None).otherwise(data.Payment_Behaviour))
data = data.na.drop()
data.show()
print('rows =',data.count())

# Handling outliers
data = data.filter((col('Num_Credit_Card') > 0) & (col('Num_Credit_Card') < 15))
data = data.filter((col('Age') > 0) & (col('Age') < 100))
data = data.filter((col('Num_Bank_Accounts') > 0) & (col('Num_Bank_Accounts') < 15))
data = data.filter((col('Interest_Rate') > 0) & (col('Interest_Rate') < 100))
data = data.filter((col('Num_of_Loan') > 0) & (col('Num_of_Loan') < 50))
data = data.filter((col('Num_Credit_Inquiries') > 0) & (col('Num_Credit_Inquiries') < 100))
data.show()
print('rows =',data.count())

from pyspark.ml.feature import StringIndexer
label_indexer = StringIndexer(inputCol='Credit_Score', outputCol='label')
data = label_indexer.fit(data).transform(data)
data.show()
print('rows =',data.count())



import pandas as pd

#PySpark DataFrame to a Pd DataFrame

#filtered_data = data.toPandas()

#Pd DataFrame to CSV file
#filtered_data.to_csv('/content/filtered.csv', index=False)



from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler, OneHotEncoder

categorical_columns = ['Occupation', 'Type_of_Loan', 'Credit_Mix','Credit_History_Age','Payment_of_Min_Amount','Payment_Behaviour']

for column in categorical_columns:
    indexer = StringIndexer(inputCol=column, outputCol=column+"_index")
    data = indexer.fit(data).transform(data)

for column in categorical_columns:
     encoder = OneHotEncoder(inputCol=column+"_index", outputCol=column+"_encoded")
     data = encoder.fit(data).transform(data)

numerical_columns = ['Age', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts','Num_Credit_Card','Interest_Rate','Num_of_Loan','Delay_from_due_date',
                  'Num_of_Delayed_Payment','Changed_Credit_Limit','Num_Credit_Inquiries','Outstanding_Debt','Credit_Utilization_Ratio','Total_EMI_per_month',
                  'Amount_invested_monthly','Monthly_Balance']
assembler_inputs = [column+"_encoded" for column in categorical_columns] + numerical_columns
vector_assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")
new_data = vector_assembler.transform(data)
std_scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
new_data = std_scaler.fit(new_data).transform(new_data)

new_data.show()
print('rows =',new_data.count())



# Splitting the data into training and testing data
(training_data, testing_data) = new_data.randomSplit([0.7, 0.3], seed=100)

training_data.show()
print('rows =',training_data.count())



### DecisionTreeClassifier ###

from pyspark.ml.classification import DecisionTreeClassifier

# Creating classifier object and appliying on training data
tree = DecisionTreeClassifier(featuresCol="scaled_features", labelCol="label")
tree_model = tree.fit(training_data)

# Creating predictions for testing data and confusion matrix
pred = tree_model.transform(testing_data)
pred.groupBy('label', 'prediction').count().show()
pred.show()

# Calculating elements of confusion matrix
TN = pred.filter('prediction = 0 AND label = prediction').count()
FN = pred.filter('prediction = 0 AND label = 1').count()
TP = pred.filter('prediction = 1 AND label = prediction').count()
FP = pred.filter('prediction = 1 AND label = 0').count()

# Measuring accuracy, precision, recall and F1
accuracy = (TP + TN) / (TP + TN + FP + FN)
print('accuracy =',accuracy)
precision = TP / (FP + TP)
print('precision =',precision)
recall = TP / (FN + TP)
print('recall =',recall)
F1 = 2 * (recall * precision) / (recall + precision)
print('F1 score =',F1)



### LogisticRegression ###

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
# Creating classifier object and appliying on training data
lr = LogisticRegression(featuresCol="scaled_features", labelCol="label").fit(training_data)

# Creating predictions for testing data and confusion matrix
pred = lr.transform(testing_data)
pred.groupBy("label", "prediction").count().show()

evaluator_accuracy = MulticlassClassificationEvaluator(metricName="accuracy", labelCol="label", predictionCol="prediction")
evaluator_F1 = MulticlassClassificationEvaluator(metricName="f1", labelCol="label", predictionCol="prediction")
evaluator_precision = MulticlassClassificationEvaluator(metricName="weightedPrecision", predictionCol="prediction", labelCol="label")
evaluator_recall = MulticlassClassificationEvaluator(metricName="weightedRecall", predictionCol="prediction", labelCol="label")

accuracy = evaluator_accuracy.evaluate(pred)
F1 = evaluator_F1.evaluate(pred)
precision = evaluator_precision.evaluate(pred)
recall = evaluator_recall.evaluate(pred)

print("Accuracy =",accuracy)
print("F1 =",F1)
print("precision =",precision)
print("recall =",recall)



### RandomForestClassifier ###

from pyspark.ml.classification import RandomForestClassifier
# Creating classifier object and appliying on training data
rf = RandomForestClassifier(numTrees= 50, featuresCol="scaled_features", labelCol="label", maxDepth=30)
model = rf.fit(training_data)

# Creating predictions for testing data and confusion matrix
pred = model.transform(testing_data)
pred.groupBy("label", "prediction").count().show()

evaluator_accuracy = MulticlassClassificationEvaluator(metricName="accuracy", labelCol="label", predictionCol="prediction")
evaluator_F1 = MulticlassClassificationEvaluator(metricName="f1", labelCol="label", predictionCol="prediction")
evaluator_precision = MulticlassClassificationEvaluator(metricName="weightedPrecision", predictionCol="prediction", labelCol="label")
evaluator_recall = MulticlassClassificationEvaluator(metricName="weightedRecall", predictionCol="prediction", labelCol="label")

accuracy = evaluator_accuracy.evaluate(pred)
F1 = evaluator_F1.evaluate(pred)
precision = evaluator_precision.evaluate(pred)
recall = evaluator_recall.evaluate(pred)

print("Accuracy =",accuracy)
print("F1 =",F1)
print("precision =",precision)
print("recall =",recall)





































numerical_columns = ['Age', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts','Num_Credit_Card','Interest_Rate','Num_of_Loan','Delay_from_due_date',
                  'Num_of_Delayed_Payment','Changed_Credit_Limit','Num_Credit_Inquiries','Outstanding_Debt','Credit_Utilization_Ratio','Total_EMI_per_month',
                  'Amount_invested_monthly','Monthly_Balance']









import matplotlib.pyplot as plt

# count the instances of each class in the label column
class_counts = new_data.groupBy('Credit_Score').count().orderBy('Credit_Score')

# convert the PySpark DataFrame to a Pandas DataFrame for plotting
class_counts_pandas = class_counts.toPandas()

# create a bar chart of the class counts
fig, ax = plt.subplots(figsize=(8, 6))
class_counts_pandas.plot(kind='bar', x='Credit_Score', y='count', ax=ax)
ax.set_xlabel('Class')
ax.set_ylabel('Count')
ax.set_title('Class Distribution')

# show the plot
plt.show()