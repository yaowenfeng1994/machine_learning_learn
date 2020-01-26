from pyspark import SparkContext

sc = SparkContext("spark://node-1:7077", "count app")
# logData = sc.textFile("file:///home/w.txt").cache()
logData = sc.textFile("hdfs://node-hadoop:9000/hello").cache()

# words = sc.parallelize(
#     ["scala",
#      "java",
#      "hadoop",
#      "spark",
#      "akka",
#      "spark vs hadoop",
#      "pyspark",
#      "pyspark and spark"
#      ])
# counts = words.count()
# print("Number of elements in RDD -> %i" % counts)

resultRdd = logData.flatMap(lambda line: line.split(" ")).map(lambda word: (word, 1)).reduceByKey(lambda a, b: a + b)
resultColl = resultRdd.collect()
for line in resultColl:
    print(line)

sc.stop()
