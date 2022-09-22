# Suspicious network event recognition
In this challenge, the task is to detect truly suspicious events and false alarms within the set of so-called network
traffic alerts that the Security Operations Center (SOC) Team members have to analyze daily.

This data set comes from IEEE BigData 2019 Cup: Suspicious Network Event Recognition challenge.
The data set available in the challenge consist of alerts investigated by a SOC team at the Security on Demand company (SoD).
It calls such signals 'investigated'. Each record is described by various statistics selected based on experts' knowledge
and a hierarchy of associated IP addresses (anonymized), called assets. For each alert in the 'investigated alerts' data 
tables, there is a history of related log events (a detailed set of network operations acquired by SoD, anonymized
to ensure the safety of SoD clients).

The data sets cover half a year between October 1, 2018, and March 31, 2019.  The main data was divided on a training set
and a test set based on alert timestamps. The training set (the file cybersecurity_training.csv) utilizes
approximately four months, and the remaining part constitutes a test set (the file cybersecurity_test.csv).
The format of those two files is the same - columns are separated by the vertical line '|' sign. However,
the target column called 'notified' is missing in the test data.

### The task:
The job is to predict which of the investigated alerts were considered truly suspicious by the SOC team and led
to issuing a notification to SoD's clients. In the training data, this information is indicated by the column 'notified'.

### Evaluation:
The AUC measure

### Tool
DataSpell

### Language
**Python**

### Libraries
sklearn, xgboost, numpy, pandas, matplotlib, pandas-profiling, seaborn  
