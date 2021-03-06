# Default Credit Data Analysis and Prediction Using Big Data
![](https://img.shields.io/badge/CODE-PYTHON-informational?style=flat&logo=<LOGO_NAME>&logoColor=white&color=2bbc8a)
![](https://img.shields.io/badge/version-3.7.3-informational?style=flat&logo=<LOGO_NAME>&logoColor=white&color=2bbc8a)
![](https://img.shields.io/badge/Cloudera-6.3.2-informational?style=flat&logo=<LOGO_NAME>&logoColor=white&color=2bbc8a)
![](https://img.shields.io/badge/Tableau-2019.4-informational?style=flat&logo=<LOGO_NAME>&logoColor=white&color=2bbc8a)
![](https://img.shields.io/badge/AWS-HDFS-informational?style=flat&logo=<LOGO_NAME>&logoColor=white&color=2bbc8a)
![](https://img.shields.io/badge/ML-Yarn-informational?style=flat&logo=<LOGO_NAME>&logoColor=white&color=2bbc8a)
![](https://img.shields.io/badge/Mapreduce-informational?style=flat&logo=<LOGO_NAME>&logoColor=white&color=2bbc8a)
![](https://img.shields.io/badge/Domain-Finance-informational?style=flat&logo=<LOGO_NAME>&logoColor=white&color=2bbc8a)
![](https://img.shields.io/badge/PySpark-informational?style=flat&logo=<LOGO_NAME>&logoColor=white&color=2bbc8a)

Credit defaulting results in a large profit loss to banks and other credit lenders. The success of the banking industry results in the ability to understand risk. Therefore, overall, the dataset chosen to analyze is a credit default analysis dataset of data from a Taiwan credit lender from the timeframe April 2005 to September 2005 and contains 25 dimensions, including Age and the predicted value for whether or not a client would be seen as defaulting. The goals of the project and analysis are to create a more efficient analysis processing system using Big Data technologies, as well as potentially improving upon defaulting prediction accuracy rate and finding additional prediction feature sets that provide high defaulting prediction model accuracies. Using Cloudera and AWS processes, such as Spark Hadoop, MapReduce and Yarn ,a Big Data System was made to conduct said processes. The feature set found with the highest rates was made up of the attributes past payment status, sex, marital status, and credit limit with the highest accuracy rate being 81.80%, acquired from the GBT with cross validation model.

![image](https://user-images.githubusercontent.com/55992728/117172340-bb5d7d80-ad99-11eb-9a91-54dd9a34b56e.png)


# Dataset Details - default of credit card clients.csv
The dataset chosen is titled ???Default of Credit Card Clients??? and looks at the default customer payments data within the timeframe April 2005 ??? September 2005. The dataset can originally be found from the UCI Machine Learning Repository  (Yeh,2016). The data was collected with the intention to predict default probability, to see which clients are credible and which are not. The dataset contains 25 attributes, comprised of categorical, integer, and real data, with 30,000 observations, with no null values. 
The dataset is comprised of the customer related attributes ID, credit limit, gender, education level, marital status, age, past payment status, bill statement, payment statement, and anticipated defaulting status, where each row contains a different customer???s status across each of the attributes. 

# Software and Big Data Technologies

Virtual Machine is used to allow for a Linux/Ubuntu platform to download and set up the Cloudera 6.3.2 connection. AWS is used to set up the hardware and software processes. HDFS is used for data storage. Spark, or specifically PySpark, is used for further analysis and machine learning on Yarn, with MapReduce as the background process for job processing and initial processing. The coding is done in PySpark which is a Python Programming Language used to integrate with Spark. Tableau and Seaborn are used for the visualizations. 

![image](https://user-images.githubusercontent.com/55992728/117172229-9a952800-ad99-11eb-933c-33626224fd39.png)
