# โค้ด PySpark สำหรับทำ Project 2 (Recommendation System)

**Dataset:** Goodbooks-10k (<https://github.com/zygmuntz/goodbooks-10k>)

ไฟล์นี้จะเป็นโค้ด PySpark ทั้งหมดตั้งแต่ต้นจนจบ สามารถนำไปรันใน Jupyter Notebook, Google Colab (ที่ลง PySpark), หรือรันบน Databricks / AWS EMR ได้เลย

---

## 1. ติดตั้งและเริ่มทำงาน (Setup & Initialize)

ถ้าใช้ Google Colab ต้องลง PySpark ก่อน:

```python
!pip install pyspark
```

เริ่มต้น `SparkSession`:

```python
from pyspark.sql import SparkSession

# สร้าง Spark Session
spark = SparkSession.builder \
    .appName("BookRecommendation_Project2") \
    .getOrCreate()
```

---

## 2. โหลดข้อมูล (Load Data)

ทำการโหลดไฟล์ข้อมูล `ratings.csv` และ `books.csv`

```python
# โหลดข้อมูล ratings ของผู้ใช้
ratings = spark.read.csv("ratings.csv", header=True, inferSchema=True)

# โหลดข้อมูลหนังสือ
books = spark.read.csv("books.csv", header=True, inferSchema=True)

# ดูตัวอย่างข้อมูล
print("--- ซับเซ็ตของ ratings ---")
ratings.show(5)

print("--- ซับเซ็ตของ books ---")
books.select("book_id", "title", "authors", "average_rating").show(5)
```

---

## 3. จัดเตรียมข้อมูล (Data Preprocessing)

ลบค่าว่าง (ถ้ามี) การรัน ALS ต้องการให้ user_id และ ข้อมูล item_id (หรือ book_id) เป็นตัวเลข, ซึ่งใน Dataset มีให้อยู่แล้ว

```python
# ลบแถวที่มีค่าว่างใน column ที่จำเป็น
ratings = ratings.dropna(subset=["user_id", "book_id", "rating"])

# เปลี่ยนชื่อคอลัมน์ให้อ่านง่าย (ถ้าจำเป็น) ไม่เปลี่ยนก็รันต่อได้
```

---

## 4. แบ่งข้อมูลสำหรับ Train และ Test

แบ่งข้อมูลเป็น Train set 80% และ Test set 20% เพื่อประเมินความแม่นยำของ Model

```python
# แบ่งข้อมูลเป็นชุดสอน (Train) 80% และชุดทดสอบ (Test) 20%
(trainingData, testData) = ratings.randomSplit([0.8, 0.2], seed=42)
```

---

## 5. สร้าง Model Recommendation แบบ Collaborative Filtering (ALS)

ใช้อัลกอริทึม **Alternating Least Squares (ALS)** ซึ่ง PySpark มีมาให้ ใช้สำหรับทายความชอบของ User (Rating)

```python
from pyspark.ml.recommendation import ALS

# กำหนด Config ให้โมเดล
als = ALS(
    maxIter=10,                      # จำนวนรอบในการ Train โมเดล
    regParam=0.1,                    # ค่าป้องกันไม่ให้โยงข้อมูลเฉพาะจุดมากเกินไป (Regularization)
    userCol="user_id",               # คอลัมน์ User
    itemCol="book_id",               # คอลัมน์ หนังสือ
    ratingCol="rating",              # คอลัมน์ คะแนน Rating
    coldStartStrategy="drop"         # หากเจอ User ใหม่หรือเล่มใหม่ใน testData ให้ drop (ป้องกันการพัง)
)

# เริ่มทำการ Train Model
model = als.fit(trainingData)
```

---

## 6. วัดผลความแม่นยำของ Model

ใช้ **RMSE (Root Mean Square Error)** หรือค่าคลาดเคลื่อน เพื่อดูว่าจาก Rating (1-5) ตีความผิดไปเท่าไหร่ (ค่ายิ่งน้อยยิ่งดี)

```python
from pyspark.ml.evaluation import RegressionEvaluator

# ให้ Model เข้าไปทายค่า rating ของชุดทดสอบ (Test)
predictions = model.transform(testData)

# ใช้ RegressionEvaluator เพื่อวัดค่า RMSE
evaluator = RegressionEvaluator(
    metricName="rmse", 
    labelCol="rating",
    predictionCol="prediction"
)

rmse = evaluator.evaluate(predictions)
print(f"Root Mean Square Error (RMSE) ของระบบแนะนำหนังสือ = {rmse:.4f}")
```

---

## 7. แนะนำหนังสือที่เหมาะสมให้ผู้ใช้งาน (Generate Recommendations)

การสุ่มแนะนำหนังสือ 5 เล่ม ให้ User แต่ละคน

```python
# แนะนำหนังสือ 5 เล่ม ให้ผู้ใช้งานแต่ละคน
userRecs = model.recommendForAllUsers(5)

# ดูข้อมูลดิบหลังประมวลผล (จะเป็น Array ข้อมูล)
print("--- คำแนะนำข้อมูลดิบของผู้ใช้ 5 คน ---")
userRecs.show(5, truncate=False)
```

**ทำให้ผลลัพธ์อ่านออกง่าย (Join คู่กับชื่อหนังสือ):**

```python
from pyspark.sql.functions import explode

# ขยาย Array ให้กลายเป็นแต่ละแถว
userRecs_exploded = userRecs.withColumn("recommendation", explode("recommendations")) \
    .select("user_id", "recommendation.book_id", "recommendation.rating")

# นำไป Join กับตาราง Books เพื่อให้เห็น "ชื่อหนังสือ" และ "ผู้แต่ง"
final_recommendations = userRecs_exploded.join(books, "book_id") \
    .select("user_id", "title", "authors", "rating") \
    .withColumnRenamed("rating", "predicted_rating") \
    .orderBy("user_id", "predicted_rating", ascending=[True, False])

print("--- รายชื่อหนังสือที่แนะนำให้ User แต่ละคน (อ่านง่าย) ---")
final_recommendations.show(20, truncate=False)
```

---

## 8. จบการทำงาน

```python
spark.stop()
```
