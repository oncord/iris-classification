from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from etl_pipeline import extract
from etl_pipeline import transform
from etl_pipeline import train
from etl_pipeline import evaluate

with DAG(
        dag_id='airflow_classification_ml_pipeline',
        description='ETL pipeline for ML classification of Iris dataset',
        schedule=None,
        start_date=datetime(2024,12,26),
        catchup=False,
) as dag:
    extract_task = PythonOperator(task_id='extract',python_callable=extract)
    transform_task = PythonOperator(task_id='transform',python_callable=transform)
    train_task = PythonOperator(task_id='train',python_callable=train)
    evaluate_task = PythonOperator(task_id='evaluate',python_callable=evaluate)

    extract_task >> transform_task >> train_task >> evaluate_task

extract_task
transform_task
train_task
evaluate_task
