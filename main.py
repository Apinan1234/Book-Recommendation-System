from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import explode
import os
import sys

# ให้รองรับภาษาไทยใน Terminal Windows
sys.stdout.reconfigure(encoding='utf-8')

print("="*60)
print("[1] กำลังเริ่มต้น Spark Session...")
print("="*60)
spark = SparkSession.builder \
    .appName("BookRecommendation_Project2") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

# กำหนดสิทธิ์ให้ log แสดงเฉพาะ ERROR จะได้ไม่เกะกะ
spark.sparkContext.setLogLevel("ERROR")

# ตรวจสอบว่ามีข้อมูลในโฟลเดอร์หรือไม่
data_dir = "goodbooks-10k"
ratings_path = os.path.join(data_dir, "ratings.csv")
books_path = os.path.join(data_dir, "books.csv")

if not os.path.exists(ratings_path):
    print(f"❌ ไม่พบโฟลเดอร์ {data_dir} กรุณา pull ข้อมูลมาลงโปรเจ็คก่อน")
    sys.exit(1)

print("\n"+"="*60)
print("[2] กำลังโหลดข้อมูล Books และ Ratings...")
print("="*60)
ratings = spark.read.csv(ratings_path, header=True, inferSchema=True)
books = spark.read.csv(books_path, header=True, inferSchema=True)

print("--- ตัวอย่างข้อมูล ratings ---")
ratings.show(3)

print("\n"+"="*60)
print("[3] เตรียมข้อมูลและแบ่ง Train / Test...")
print("="*60)
ratings = ratings.dropna(subset=["user_id", "book_id", "rating"])
(trainingData, testData) = ratings.randomSplit([0.8, 0.2], seed=42)
print("-> แบ่งข้อมูลเสร็จสิ้น: Train 80%, Test 20%")

print("\n"+"="*60)
print("[4] กำลัง Train Model Recommendation (ALS)... อาจใช้เวลาสักครู่...")
print("="*60)
als = ALS(maxIter=10, regParam=0.1, userCol="user_id", itemCol="book_id", ratingCol="rating", coldStartStrategy="drop")
model = als.fit(trainingData)
print("-> Train Model สำเร็จ!")

print("\n"+"="*60)
print("[5] กำลังประเมินผลความแม่นยำด้วย RMSE...")
print("="*60)
predictions = model.transform(testData)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print(f"✅ Root Mean Square Error (RMSE) ของระบบแนะนำ = {rmse:.4f}")

print("\n"+"="*60)
print("[6] แสดงผลการแนะนำหนังสือให้ 5 ผู้ใช้ (Top 5 Recommendations)...")
print("="*60)

# ลองดึงข้อมูลคำแนะนำให้ user 5 คนแรกเพื่อความรวดเร็วในการแสดง
users = ratings.select("user_id").distinct().limit(5)
userRecs = model.recommendForUserSubset(users, 5)

# แกะ Array ข้อมูลเพื่อให้ Join กับตารางหนังสือได้
userRecs_exploded = userRecs.withColumn("recommendation", explode("recommendations")) \
    .select("user_id", "recommendation.book_id", "recommendation.rating")

final_recommendations = userRecs_exploded.join(books, "book_id") \
    .select("user_id", "title", "authors", "rating") \
    .withColumnRenamed("rating", "predicted_rating") \
    .orderBy("user_id", "predicted_rating", ascending=[True, False])

final_recommendations.show(25, truncate=False)

print("\n🎉 รันเสร็จสมบูรณ์! หยุดการทำงานของ Spark...")
spark.stop()
