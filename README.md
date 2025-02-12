# **ETL Pipeline for Iris classification**

## **Overview**

This project was inspired by another project ([326_Final_Project.pdf](https://github.com/user-attachments/files/18763562/326_Final_Project.pdf)) done in my probability/statistics with R class in which we performed an EDA/IDA on R's built-in Iris dataset and determined which feature(s) of the Iris flower were significant in classifying the species of Iris flower.
In this project, I construct a very simple batch ETL data pipeline to load a ML classification model using Apache Airflow to orchestrate the downstream tasks of ETL.
The use of Apache Airflow in this context is quite overkill and is only intended to be a proof of concept of automating the ETL process, along with acting as a learning experience for DE/ML tools/principles.

## **Notes on model**

I decided to use the Random Forest Algorithm to classify each species of Iris flower since an ensemble approach would enable decision trees to search though every possible feature to determine the most significant feature in aggregate.

