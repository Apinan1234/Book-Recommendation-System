# รายงาน Project 2: Book Recommendation System

**Dataset:** Goodbooks-10k

---

## 1. จะทำการ Recommend อะไร?

โปรเจกต์นี้จะทำการ **พัฒนาระบบแนะนำหนังสือ (Book Recommendation System)** ที่สามารถพยากรณ์และแนะนำ "หนังสือเล่มใหม่ที่ผู้ใช้งานแต่ละคนน่าจะชอบและให้คะแนนสูง" โดยวิเคราะห์จากประวัติการให้คะแนน (Rating) ของผู้ใช้ในอดีต (Collaborative Filtering)

---

## 2. อธิบายข้อมูลที่นำมาใช้ (Dataset)

ข้อมูลที่ใช้คือ **Goodbooks-10k** ประกอบด้วยข้อมูลหนังสือยอดนิยม 10,000 เล่ม และประวัติการให้คะแนนจากผู้ใช้มากกว่า 6 ล้านรายการ ซึ่งแบ่งเป็น 2 ตารางหลักที่สำคัญ ได้แก่:

### 2.1 ตารางคะแนน (ratings.csv)

เป็นข้อมูลที่รวบรวมว่าผู้ใช้แต่ละคนให้คะแนนหนังสือแต่ละเล่มเท่าไหร่ ประกอบด้วย Field หลักคือ:

- `user_id`: รหัสประจำตัวของผู้ใช้
- `book_id`: รหัสประจำตัวของหนังสือ
- `rating`: คะแนนที่ผู้ใช้ให้กับหนังสือเล่มนั้น ๆ (มีค่า 1 - 5 ดาว)

### 2.2 ตารางข้อมูลหนังสือ (books.csv)

เป็นข้อมูลรายละเอียดของหนังสือแต่ละเล่มเพื่อใช้ในการอธิบายผลลัพธ์ ประกอบด้วย Field สำคัญคือ:

- `book_id`: รหัสประจำตัวของหนังสือ
- `title`: ชื่อหนังสือ
- `authors`: ชื่อผู้แต่งหนังสือ
- `average_rating`: คะแนนรีวิวเฉลี่ยโดยรวมของหนังสือเล่มนี้
- `ratings_count`: จำนวนคนที่มารีวิวหนังสือเล่มนี้

---

## 3. การเตรียมการและติดตั้งเครื่องมือ (Setup)

เพื่อรันระบบ Book Recommendation System จำเป็นต้องติดตั้งไลบรารีที่เกี่ยวข้องผ่าน Terminal ดังนี้:

1. **PySpark**: เครื่องมือหลักสำหรับการจัดการ Big Data แบบ Distributed Computing และเทรนโมเดล Machine Learning

   ```bash
   pip install pyspark
   ```

2. **NumPy**: เครื่องมือคำนวณคณิตศาสตร์เบื้องหลังที่ PySpark (MLlib) จำเป็นต้องเรียกใช้

   ```bash
   pip install numpy
   ```

---

## 4. ขั้นตอนการทำงานทั้งหมดและ Code

ขั้นตอนในการใช้ PySpark พัฒนาระบบแนะนำหนังสือประกอบไปด้วย 5 ขั้นตอนดังนี้:

**ขั้นตอนที่ 1. การนำเข้าข้อมูล (Data Loading):**
เริ่มต้นด้วยการสร้าง Spark Environment และโหลดข้อมูลจากไฟล์ `ratings.csv` และ `books.csv`

```python
spark = SparkSession.builder.appName("BookRec").getOrCreate()
ratings = spark.read.csv("ratings.csv", header=True, inferSchema=True)
books = spark.read.csv("books.csv", header=True, inferSchema=True)
```

**ขั้นตอนที่ 2. แบ่งข้อมูล Train/Test Data:**
แบ่งข้อมูลคะแนนรีวิวออกเป็น Train set 80% เพื่อนำมาเป็นแบบฝึกหัดสอนโมเดล และ Test set 20% สำหรับเก็บไว้เป็นข้อสอบวัดความแม่นยำ (เป็นการปกปิดข้อมูลไว้ไม่ให้ AI เห็นก่อน)

```python
(trainingData, testData) = ratings.randomSplit([0.8, 0.2], seed=42)
```

**ขั้นตอนที่ 3. สร้างระบบและ Train Model (ALS Model Building):**
เราเลือกใช้อัลกอริทึม **ALS (Alternating Least Squares)** ซึ่งเป็นการทำ Recommendation แบบ Collaborative Filtering คือการจับคู่จุดร่วมความชอบแฝง (Latent Features) ของผู้ใช้ที่ชอบอะไรคล้ายกัน

มีการตั้งค่าพารามิเตอร์สำคัญระดับลึกดังนี้:

- `maxIter=10`: วนรอบการเรียนรู้ 10 รอบ เพื่อหาความสมดุลและความสัมพันธ์ที่เสถียรที่สุด
- `regParam=0.1`: ป้องกันไม่ให้โมเดลจำข้อสอบมาตอบมากเกินไป (Overfitting)
- `coldStartStrategy="drop"`: เป็นการดักจับ Error โดยตั้งค่าให้ตัดข้อมูลผู้ใช้ใหม่ (ที่โผล่มามีในช่อง Test แต่ไม่เคยมีในช่อง Train) ออกชั่วคราว ป้องกันไม่ให้โปรแกรมหยุดทำงานกะทันหัน

```python
als = ALS(maxIter=10, regParam=0.1, userCol="user_id", itemCol="book_id", ratingCol="rating", coldStartStrategy="drop")
model = als.fit(trainingData)
```

**ขั้นตอนที่ 4. วัดผลความแม่นยำ (Evaluation):**
ให้ตัวแบบโมเดลทำนายกับ Test Data ส่วนที่เหลือ แล้วหาค่า Root Mean Square Error (RMSE) เพื่อดูว่าการทำนายคะแนนคลาดเคลื่อนกี่ดาว โดยในการทดสอบพบค่าคลาดเคลื่อน (RMSE) ที่ยอมรับได้

```python
predictions = model.transform(testData)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print(f"RMSE = {rmse}")
```

**ขั้นตอนที่ 5. แนะนำหนังสือ (Generate Recommendations):**
สั่งให้ Model ทำการแนะนำหนังสือ 5 เล่มแรกที่เหมาะกับ User แต่ละรายที่สุด (คะแนนทำนายสูงที่สุด) และนำมา Join กับตาราง `books` เพื่อโชว์ชื่อหนังสือ

```python
userRecs = model.recommendForAllUsers(5)
# ... ทำการแกะ Array ด้วย explod แล้ว Join กลับเพื่อแสดงรายชื่อหนังสือ ...
final_recommendations = userRecs_exploded.join(books, "book_id").select("user_id", "title", "authors", "rating")
```

---

## 5. ผลลัพธ์ที่ได้

จากการทำงานของ PySpark

1. **ด้านความแม่นยำ**: โปรแกรมสามารถประมวลผลข้อมูลระดับ 6 ล้าน Record ได้สำเร็จด้วยเทคโนโลยีการกระจายประมวลผล (Spark) และได้ค่าความคลาดเคลื่อน (RMSE) ออกมา ซึ่งแปลผลได้ว่าตัวแบบทายคะแนนหนังสือ 1-5 ดาวพลาดไปเฉลี่ยประมาณดาวน์เรต (โดยเฉลี่ยประมาณ 0.82-0.9 ดาว) ซึ่งถือว่ามีประสิทธิภาพดี
2. **ด้านคำแนะนำช่วยผู้ใช้งาน**: โมเดลสามารถสร้างรายการ (List) แนะนำหนังสือเฉพาะเจาะจงรายคนได้สำเร็จ (Personalized Recommendation)

**[ตัวอย่างผลลัพธ์]**
*เมื่อวิเคราะห์ User รหัส 1 พบว่าหนังสือ 3 อันดับแรกที่ระบบแนะนำ ได้แก่:*

1. The Calvin and Hobbes Tenth Anniversary Book (ผู้แต่ง: Bill Watterson) คาดการณ์ว่าจะได้คะแนน 5.0
2. The Complete Calvin and Hobbes (ผู้แต่ง: Bill Watterson) คาดการณ์ว่าจะได้คะแนน 4.9
3. Harry Potter Boxset (ผู้แต่ง: J.K. Rowling) คาดการณ์ว่าจะได้คะแนน 4.8
*(ข้อมูลจากการรันตัวอย่าง)*
