import os
import sys

spark_name = os.environ.get('SPARK_HOME', None)
if not spark_name:
    raise Exception('spark环境没有配置好')
sys.path.insert(0, os.path.join(spark_name, '/Users/klook/Documents/spark-2.4.4-bin-hadoop2.7/python'))
sys.path.insert(0, os.path.join(spark_name,
                                '/Users/klook/Documents/spark-2.4.4-bin-hadoop2.7/python/lib/py4j-0.10.7-src.zip'))
exec(open(os.path.join(spark_name, 'python/pyspark/shell.py')).read())
