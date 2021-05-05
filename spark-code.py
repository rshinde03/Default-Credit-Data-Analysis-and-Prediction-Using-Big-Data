#!/usr/bin/env python
# coding: utf-8

# In[82]:


#import findspark
#findspark.init()
import pyspark as ps
import warnings
from pyspark.sql import SQLContext


# In[83]:


from pyspark.sql import DataFrameNaFunctions
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import Binarizer
from pyspark.sql.functions import avg

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split
from collections import Counter


# In[84]:


from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql import SparkSession


# In[85]:


from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split
from collections import Counter
import pandas as pds
import numpy as np
from pyspark.sql.functions import isnan, when, count, col
import seaborn as sb 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import matplotlib.pyplot as plt


# In[86]:


spark = SparkSession.builder.master("yarn").getOrCreate()


# ## Reading data file

# In[87]:


csv = spark.read.csv('/user/root/data.csv', inferSchema=True, header=True)
csv.show(5)


# ## Rename column PAY_0 as PAY_1

# In[7]:


new= csv.withColumnRenamed("PAY_0","PAY_1")


# In[8]:


new.show(5)


# ## Check for Null Values

# In[9]:


from pyspark.sql.functions import isnan, when, count, col


# In[10]:


new.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in new.columns]).show()


# **data has no NULL values in any columns**

# ## Check for Duplicates 

# Only ID column needs to be checked because each of those values should be unique

# In[11]:


if new.count() > new.dropDuplicates(['ID']).count():
    raise ValueError('There are ID duplicates within the dataset')
else:
    print('There are no duplicates')


# ## Drop ID Column

# In[12]:


new=new.drop('ID')


# In[13]:


new.show(5)


# ## Counts of each Label in dataset

# In[14]:


new.cube("default payment next month").count().show()


# ## SMOTE technique to help with the Unbalanced Default

# In[15]:


res1 = new.select("*").toPandas()


# In[16]:


res1 


# In[17]:


res1.columns.values


# In[18]:


res2 = res1[['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_1',
           'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1',
           'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
           'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 
            'PAY_AMT6']]


# In[19]:


res3 = res1['default payment next month']


# In[20]:


# filter/separate the DF into two groups
X1 = res2
Y1 = res1[['default payment next month']]


# In[21]:


X1.shape, Y1.shape


# In[22]:


trainX1, testX1, trainY1, testY1 = train_test_split(X1, Y1, 
                                                    test_size = 0.1,
                                                   random_state = 0)


# In[23]:


trainX1.columns, trainY1.columns


# In[24]:


smot1 = SMOTE(random_state = 14, ratio = 'auto', kind = 'regular')
re_trainX1, re_trainY1 = smot1.fit_sample(trainX1, trainY1)
print('The dataset resampled: {}'.format(Counter(re_trainY1)))
#need to figure out how to get a non warning filled output for the above line
#resampling output: (1: 21004, 0: 21004)


# In[25]:


sb.countplot(res3).set_title("Customer Count: Defaulting or Not prior to SMOTE (Oversampling) ")
plt.xlabel("Not Defaulting (0) versus Defaulting (1)")
plt.ylabel("Customer Count")
plt.show()


# In[26]:


sb.countplot(re_trainY1).set_title("Customer Count: Defaulting or Not post SMOTE (Oversampling) ")
plt.xlabel("Not Defaulting (0) versus Defaulting (1)")
plt.ylabel("Customer Count")
plt.show()


# After doing SMOTE, the dataset no longer has an uneven set of values

# #### Now need to convert pandas dataframe back into the spark dataframe

# In[27]:


res1.columns.values


# In[28]:


dfConv1 = pds.DataFrame(re_trainX1, columns = ['LIMIT_BAL', 'SEX',
                                              'EDUCATION', 'MARRIAGE', 
                                              'AGE', 'PAY_1', 'PAY_2', 'PAY_3',
                                              'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1',
                                               'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4',
                                              'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
                                              'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 
                                              'PAY_AMT5', 'PAY_AMT6'])
dfConv2 = pds.DataFrame(re_trainY1, columns = ['default payment next month'])
res2 = dfConv1.combine_first(dfConv2)


# In[29]:


new = spark.createDataFrame(res2)
new.show(1)


# In[30]:


new.cube("default payment next month").count().show()


# Default (1) and Not Default (0) have even counts within the set - SMOTE was successful

# ## Descriptive statistics for all columns 

# In[31]:


new.describe().show()


# ## Statistical check for Target column

# In[32]:


new.select('default payment next month').describe().show()


# ## Statistical check for BILL column

# In[33]:


new.select('BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6').describe().show()


# ## Statistical check for Pay_Amt column

# In[34]:


new.select('PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6').describe().show()


# In[35]:


new.printSchema()


# ## Convert negative values to positive

# In[36]:


from  pyspark.sql.functions import abs                                                  


# In[37]:


#new = new.withColumn('BILL_AMT1',abs(new.BILL_AMT1))
#new = new.withColumn('BILL_AMT2',abs(new.BILL_AMT2))
#new = new.withColumn('BILL_AMT3',abs(new.BILL_AMT3))
#new = new.withColumn('BILL_AMT4',abs(new.BILL_AMT4))
#new = new.withColumn('BILL_AMT5',abs(new.BILL_AMT5))
#new = new.withColumn('BILL_AMT6',abs(new.BILL_AMT6))


# ## Correlation Heat Map

# In[38]:


import numpy as np
import pandas as pd


# In[39]:


converted=new.select("*").toPandas()


# In[40]:


import seaborn as sb 


# In[41]:


res2 = converted[['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_1',
           'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1',
           'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
           'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 
            'PAY_AMT6','default payment next month']]


# In[42]:


dat1 = res2.corr()


# In[43]:


plt.figure(figsize = (14,14))
#plt.ylim(c, d)

a2 = sb.heatmap(dat1, annot = True, linecolor = 'Black',
                linewidths = 2, square = True, cmap="BuPu",
               fmt = '0.2g')


# #### log value

# In[44]:


res2


# In[45]:


#done so I don't have to keep going and rerun a bunch of lines if I mess up the value of res2
res21 = res2


# ##### taking the log of the numbers for both BILL_AMT_ and for PAY_AMT_ is going to create errors. We can do two things to help combat this.
# ##### 1. replace 0 with a very small value (maybe 0.00001)
# ##### 2. after computing the log, to fill in inf values and N/A values with 0
# ##### will try replacing 0's with the small value 

# In[46]:


#res21['BILL_AMT1'] = res21["BILL_AMT1"].replace(to_replace 0, 
#                 value = 0.00001) 
res21['BILL_AMT1'].replace({0: 0.0000001}, inplace=True)
res21['BILL_AMT2'].replace({0: 0.0000001}, inplace=True)
res21['BILL_AMT3'].replace({0: 0.0000001}, inplace=True)
res21['BILL_AMT4'].replace({0: 0.0000001}, inplace=True)
res21['BILL_AMT5'].replace({0: 0.0000001}, inplace=True)
res21['BILL_AMT6'].replace({0: 0.0000001}, inplace=True)

res21['PAY_AMT1'].replace({0: 0.0000001}, inplace=True)
res21['PAY_AMT2'].replace({0: 0.0000001}, inplace=True)
res21['PAY_AMT3'].replace({0: 0.0000001}, inplace=True)
res21['PAY_AMT4'].replace({0: 0.0000001}, inplace=True)
res21['PAY_AMT5'].replace({0: 0.0000001}, inplace=True)
res21['PAY_AMT6'].replace({0: 0.0000001}, inplace=True)


# In[47]:



res21['BILL_AMT1_LOG'] = np.log10(res2['BILL_AMT1'])
res21['BILL_AMT2_LOG'] = np.log10(res2['BILL_AMT2'])
res21['BILL_AMT3_LOG'] = np.log10(res2['BILL_AMT3'])
res21['BILL_AMT4_LOG'] = np.log10(res2['BILL_AMT4'])
res21['BILL_AMT5_LOG'] = np.log10(res2['BILL_AMT5'])
res21['BILL_AMT6_LOG'] = np.log10(res2['BILL_AMT6'])

res21['PAY_AMT1_LOG'] = np.log10(res2['PAY_AMT1'])
res21['PAY_AMT2_LOG'] = np.log10(res2['PAY_AMT2'])
res21['PAY_AMT3_LOG'] = np.log10(res2['PAY_AMT3'])
res21['PAY_AMT4_LOG'] = np.log10(res2['PAY_AMT4'])
res21['PAY_AMT5_LOG'] = np.log10(res2['PAY_AMT5'])
res21['PAY_AMT6_LOG'] = np.log10(res2['PAY_AMT6'])

pd.options.mode.chained_assignment = None 

res21


# #### Additional Heatmap

# In[48]:


res3 = res2[['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_1',
           'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1_LOG',
           'BILL_AMT2_LOG', 'BILL_AMT3_LOG', 'BILL_AMT4_LOG', 'BILL_AMT5_LOG', 'BILL_AMT6_LOG',
           'PAY_AMT1_LOG', 'PAY_AMT2_LOG', 'PAY_AMT3_LOG', 'PAY_AMT4_LOG', 'PAY_AMT5_LOG', 
            'PAY_AMT6_LOG','default payment next month']]


# In[49]:


res3


# In[50]:


dat2 = res3.corr()


# In[51]:


plt.figure(figsize = (14,14))
#plt.ylim(c, d)

a3 = sb.heatmap(dat2, annot = True, linecolor = 'Black',
                linewidths = 2, square = True, cmap="BuPu",
               fmt = '0.2g')


# ###### it marginally improved but not really by a lot

# Best attributes seem to be (setting at a cutoff of 0.22 or greater):
# 'SEX', 'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5'
# 
# if cutting off at 0.21 or higher: include 'MARRIAGE', 'PAY_6'

# In[52]:


new = spark.createDataFrame(res3)
new.show(1)
#does not want to work because of the false values that have appeared


# ## Create feature  column

# In[53]:


from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler


# Reminder: Best attributes seem to be (setting at a cutoff of 0.22 or greater):
# 'SEX', 'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5'
# 
# if cutting off at 0.21 or higher: include 'MARRIAGE', 'PAY_6'

# In[54]:


# FIRST TIME
#assembler = VectorAssembler(
#    inputCols=["LIMIT_BAL","PAY_1", "PAY_2","PAY_3","PAY_4","PAY_5","PAY_6","BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6"],
#    outputCol="features")
# keeping this in here in case the model is actually worse after


# In[55]:


# SECOND TIME - CUTOFF BEING 0.21 OR GREATER
#assembler = VectorAssembler(
#    inputCols=['SEX', 'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'MARRIAGE'],
#    outputCol="features")


# In[56]:


#THIRD TIME - CUTOFF BEING 0.22 OR GREATER FROM HEATMAP
assembler = VectorAssembler(
    inputCols=['SEX', 'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5'],
    outputCol="features")


# In[57]:


output = assembler.transform(new)


# In[58]:


output.printSchema()


# In[59]:


output.show(5)


# In[ ]:





# ## Pearson and Spearman's Correlation

# In[60]:


from pyspark.ml.stat import Correlation


# In[61]:


r2 = Correlation.corr(output, "features", "spearman").head()
print("Spearman correlation matrix:\n" + str(r2[0]))


# In[62]:


r1 = Correlation.corr(output, "features","pearson").head()
print("Pearson correlation matrix:\n" + str(r1[0]))


# ## Chi-Squared Test for feature selection

# In[66]:


from pyspark.ml.feature import ChiSqSelector
from pyspark.ml.linalg import Vectors


# In[67]:


selector = ChiSqSelector(numTopFeatures=10, featuresCol="features",
                         outputCol="selectedFeatures", labelCol="default payment next month")


# In[68]:


result = selector.fit(output).transform(output)


# In[69]:


result.show(5)


# ## Create Label column

# In[73]:


from pyspark.ml.feature import StringIndexer


# In[74]:


indexer = StringIndexer(inputCol="default payment next month", outputCol="label")


# In[75]:


labelDf = indexer.fit(result).transform(result)


# In[76]:


labelDf.printSchema()


# In[78]:


labelDf.show(5)


# In[79]:


labelDf=labelDf.select('features','label')


# In[80]:


labelDf.show(5)


# ## Test-Train split

# In[81]:


train, test = labelDf.randomSplit([0.7, 0.3], seed = 2018)
print("Training Dataset Count: " + str(train.count()))
print("Test Dataset Count: " + str(test.count()))


# ## Logistic Regression

# In[323]:


from pyspark.ml.classification import LogisticRegression


# In[324]:


lr = LogisticRegression(maxIter=10, regParam=0.3, featuresCol="features",labelCol="label" ,elasticNetParam=0.8)


# In[325]:


lrModel = lr.fit(train)


# In[326]:


prediction = lrModel.transform(test)


# In[327]:


from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator()
print('Test Area Under ROC', evaluator.evaluate(prediction))


# ORIGINAL GAVE 0.5; all log regression is giving a value of 0.5

# ## cross validation for Logistic Regression model

# In[329]:


from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


# In[330]:


paramGrid = ParamGridBuilder()     .addGrid(lr.regParam, [0.1,0.01])     .addGrid(lr.fitIntercept, [False, True])     .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])     .build()


# In[331]:


crossval = CrossValidator(estimator=lr,
                          estimatorParamMaps=paramGrid,
                          evaluator=BinaryClassificationEvaluator(),
                          numFolds=2) 


# In[332]:


cvModel= crossval.fit(labelDf)


# In[333]:


cvModel.avgMetrics


# In[334]:


predictionDf = cvModel.transform(test)


# In[335]:


predictionDf.show(5)


# In[336]:


# Print the coefficients and intercept for logistic regression
print("Coefficients: " + str(lrModel.coefficients))
print("Intercept: " + str(lrModel.intercept))


# In[337]:


from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator()
print('Test Area Under ROC', evaluator.evaluate(predictionDf))


# In[338]:


accuracy = evaluator.evaluate(predictionDf)
print(accuracy*100)


# original gave 71.25241443260926
# 
# 2nd time:75.86197534845145 -> 
# 2nd time predictors: 'SEX', 'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'MARRIAGE'
# 
# 3rd time: 74.00576341971987 -> it got worse
# 3rd time predictors: 'SEX', 'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5'

# ## Random Forest with cross validation

# In[257]:


from pyspark.ml.classification import RandomForestClassifier
rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'label', numTrees=100)
rfModel = rf.fit(train)
predictions = rfModel.transform(test)


# In[258]:


crossval_rf = CrossValidator(estimator=rf,
                          estimatorParamMaps=paramGrid,
                          evaluator=BinaryClassificationEvaluator(),
                          numFolds=2) 


# In[259]:


cvModel_rf= crossval_rf.fit(labelDf)


# In[260]:


cvModel_rf.avgMetrics


# In[261]:


predictionDf_rf = cvModel_rf.transform(test)


# In[262]:


evaluator = BinaryClassificationEvaluator()
print("Test Area Under ROC: " + str(evaluator.evaluate(predictionDf_rf, {evaluator.metricName: "areaUnderROC"})))


# original gave Test Area Under ROC: 0.7969682070997004 - with original predictors
# 
# 2nd time with new predictors: Test Area Under ROC: 0.7995916177367415
# 2nd time predictors: 'SEX', 'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'MARRIAGE'
# 
# Now: Try improving model with tuning hyper parameters

# ## Random Forest with cross validation 
# ####  try: numFolds = 5 

# In[339]:


from pyspark.ml.classification import RandomForestClassifier
rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'label', numTrees=100)
rfModel = rf.fit(train)
predictions = rfModel.transform(test)


# In[340]:


crossval_rf = CrossValidator(estimator=rf,
                          estimatorParamMaps=paramGrid,
                          evaluator=BinaryClassificationEvaluator(),
                          numFolds=5) 


# In[341]:


cvModel_rf= crossval_rf.fit(labelDf)


# In[342]:


cvModel_rf.avgMetrics


# In[343]:


predictionDf_rf = cvModel_rf.transform(test)


# In[345]:


evaluator = BinaryClassificationEvaluator()
print("Test Area Under ROC: " + str(evaluator.evaluate(predictionDf_rf, {evaluator.metricName: "areaUnderROC"})))


# original gave Test Area Under ROC: 0.7969682070997004
# 
# Marginally improved second time with: 0.7995916177367415 
# Increasing number of folds gave the exact same value: Test Area Under ROC: 0.7995916177367415
# 
# 3rd time with the predictors >= 0.22: 0.7901307429820612 - got marginally worse
# I think the heatmap predictors of >= 0.22 is starting to be overfitted

# In[ ]:





# ## GBT Classifier

# In[272]:


from pyspark.ml.classification import GBTClassifier
gbt = GBTClassifier(maxIter=10)
gbtModel = gbt.fit(train)
predictions = gbtModel.transform(test)


# In[273]:


evaluator = BinaryClassificationEvaluator()
print("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))


# In[274]:


crossval_gbt = CrossValidator(estimator=gbt,
                          estimatorParamMaps=paramGrid,
                          evaluator=BinaryClassificationEvaluator(),
                          numFolds=2) 


# In[275]:


cvModel_gbt= crossval_gbt.fit(labelDf)


# In[276]:


cvModel_gbt.avgMetrics


# In[277]:


predictionDf_gbt = cvModel_gbt.transform(test)


# In[278]:


evaluator = BinaryClassificationEvaluator()
print("Test Area Under ROC: " + str(evaluator.evaluate(predictionDf_gbt, {evaluator.metricName: "areaUnderROC"})))


# In[279]:


#original gave Test Area Under ROC: 0.8277322240593465

#wow... it got reduced... just marginally but... great: Test Area Under ROC: 0.8217262618899654


# ## Decision tree

# In[1]:


from pyspark.ml.classification import DecisionTreeClassifier
dt = DecisionTreeClassifier(featuresCol = 'features', labelCol = 'label', maxDepth = 3)
dtModel = dt.fit(train)
predictions = dtModel.transform(test)


# In[281]:


crossval_d = CrossValidator(estimator=dt,
                          estimatorParamMaps=paramGrid,
                          evaluator=BinaryClassificationEvaluator(),
                          numFolds=2) 


# In[282]:


cvModel_d= crossval_d.fit(train)


# In[283]:


cvModel_d.avgMetrics


# In[284]:


predictionDf_d = cvModel_d.transform(test)


# In[285]:


evaluator = BinaryClassificationEvaluator()
print("Test Area Under ROC: " + str(evaluator.evaluate(predictionDf_d, {evaluator.metricName: "areaUnderROC"})))


# orginially: gave Test Area Under ROC: 0.6185580237403748
# 
# 2nd time w/ predictors: Test Area Under ROC: 0.7064587753553144

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




