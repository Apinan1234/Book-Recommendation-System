# รายงาน Project 2: Book Recommendation System

**Data :** Goodbooks-10k dataset ([https://github.com/zygmuntz/goodbooks-10k](https://github.com/zygmuntz/goodbooks-10k))

**อธิบาย Data :**
ชุดข้อมูล Goodbooks-10k ประกอบด้วยข้อมูลหนังสือยอดนิยม 10,000 เล่ม และประวัติการให้คะแนน (Rating) จากผู้ใช้มากกว่า 6 ล้านรายการ
ชุดข้อมูลนี้เหมาะสำหรับการพัฒนาระบบแนะนำหนังสือ (Book Recommendation System) เพื่อพยากรณ์และแนะนำหนังสือเล่มใหม่ที่ผู้ใช้แต่ละคนน่าจะชื่นชอบ โดยอ้างอิงจากประวัติลักษณะการให้คะแนนความชอบในอดีตของผู้ใช้คนนั้น ๆ ร่วมกับผู้ใช้อื่นที่มีรสนิยมคล้ายคลึงกัน (Collaborative Filtering)

**Field Description :**
ชุดข้อมูลที่เลือกนำมาวิเคราะห์แบ่งออกเป็น 2 ตารางหลักที่สำคัญ ได้แก่:

1. **ตาราง `ratings.csv` (ตารางพฤติกรรมการให้คะแนน)**

   - `user_id`: รหัสประจำตัวของผู้ใช้งาน (ตัวเลข)
   - `book_id`: รหัสประจำตัวของหนังสือที่ผู้ใช้งานเคยให้คะแนน (ตัวเลข)
   - `rating`: ระดับคะแนนที่ผู้ใช้ให้กับหนังสือเล่มนั้น ๆ (มีค่าตั้งแต่ 1 ถึง 5 ดาว)
2. **ตาราง `books.csv` (ตารางรายละเอียดข้อมูลหนังสือ)**

   - `book_id`: รหัสประจำตัวของหนังสือ (เพื่อใช้ระบุและจับคู่กับตาราง Ratings)
   - `title`: ชื่อหนังสือ
   - `authors`: ชื่อผู้แต่งหนังสือ
   - `average_rating`: อัตราส่วนคะแนนรีวิวเฉลี่ยโดยรวมทั้งหมดของหนังสือเล่มนี้
   - `ratings_count`: จำนวนผู้ใช้ทั้งหมดที่มารีวิวหรือให้คะแนนหนังสือเล่มนี้

---

**การวิเคราะห์ด้วย :**
ใช้ **PySpark (Apache Spark)** ในการทำ Distributed Data Processing
และสร้างโมเดล Machine Learning ด้วยอัลกอริทึม **Alternating Least Squares (ALS)** ซึ่งเป็นเทคนิคประเภท **Collaborative Filtering**

**ขั้นตอน :**

1. **การกำหนดค่าเริ่มต้น (Setup):** สร้าง Spark Environment (SparkSession) เพื่อรองรับการทำงานกับข้อมูลขนาดใหญ่แบบ Distributed
2. **การนำเข้าข้อมูล (Data Loading):** โหลดข้อมูลในรูปแบบ DataFrame จากไฟล์ `ratings.csv` และ `books.csv`
3. **การล้างข้อมูล (Data Cleaning):** กรองและลบข้อมูลที่เป็นค่าว่าง (Null) ออกจากคอลัมน์ที่จำเป็นต่อการเอาไปประมวลผล (`user_id`, `book_id`, `rating`)
4. **การแบ่งชุดข้อมูล (Train/Test Split):** แบ่งข้อมูลออกเป็น 2 ส่วน คือ Train set (80%) เพื่อให้ระบบเรียนรู้และสอนโมเดล และ Test set (20%) เพื่อเก็บไว้สอบวัดความแม่นยำหลังนำไปเทรนเสร็จ
5. **การสร้างและสอนโมเดล (Model Building):** นำข้อมูล Train set ป้อนเข้าสู่เทคนิค Collaborative Filtering โดยตั้งค่าพารามิเตอร์อัลกอริทึม ALS ให้ทำงาน 10 รอบในการหากลุ่มผู้ใช้ที่มีรสนิยมคล้ายคลึงกัน
6. **การวัดผล (Evaluation):** ตรวจสอบและประเมินความแม่นยำของโมเดลด้วยค่าสถิติ RMSE (Root Mean Square Error) เพื่อประเมินค่าความคลาดเคลื่อนของการทายระดับดาว
7. **การสร้างผลลัพธ์คำแนะนำ (Generate Recommendations):** สั่งโมเดลแนะนำหนังสือที่เหมาะสมที่สุด 5 อันดับแรกให้กับผู้ใช้งานแต่ละคน (Top-5 Recommendation) จากนั้นเชื่อมโยงข้อมูล (Join) กลับไปที่ข้อมูลหลักเพื่อดึงแสดงชื่อหนังสือ

---

**CODE :**

```python
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import explode
import os

# 1. กำหนดค่าเริ่มต้นและสร้าง SparkSession
spark = SparkSession.builder \
    .appName("BookRecommendation_Project2") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# 2. นำเข้าข้อมูล
data_dir = r"c:\Users\USER\OneDrive\Documents\BEAM\ANN\goodbooks-10k"
ratings = spark.read.csv(os.path.join(data_dir, "ratings.csv"), header=True, inferSchema=True)
books = spark.read.csv(os.path.join(data_dir, "books.csv"), header=True, inferSchema=True)

# 3. ล้างข้อมูล (ลบดรอปข้อมูลรายการที่มีค่าว่าง)
ratings = ratings.dropna(subset=["user_id", "book_id", "rating"])

# 4. แบ่งข้อมูลสำหรับ Train 80% และ Test 20%
(trainingData, testData) = ratings.randomSplit([0.8, 0.2], seed=42)

# 5. สร้าง Model Recommendation ด้วย ALS Algorithm 
als = ALS(maxIter=10, regParam=0.1, userCol="user_id", itemCol="book_id", 
          ratingCol="rating", coldStartStrategy="drop")
model = als.fit(trainingData)

# 6. ประเมินผลความแม่นยำเบื้องต้นด้วย Root Mean Square Error (RMSE)
predictions = model.transform(testData)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print(f"Root Mean Square Error (RMSE) = {rmse:.4f}")

# 7. แนะนำหนังสือ 5 เล่มให้ผู้ใช้จำลองจำนวน 5 คน (Top 5 Recommendations)
users = ratings.select("user_id").distinct().limit(5)
userRecs = model.recommendForUserSubset(users, 5)

# แกะโครงสร้าง Array เพื่อให้แสดงผลลัพธ์ให้อ่านเข้าใจง่ายคู่กับชื่อหนังสือ
userRecs_exploded = userRecs.withColumn("recommendation", explode("recommendations")) \
    .select("user_id", "recommendation.book_id", "recommendation.rating")

final_recommendations = userRecs_exploded.join(books, "book_id") \
    .select("user_id", "title", "authors", "rating") \
    .withColumnRenamed("rating", "predicted_rating") \
    .orderBy("user_id", "predicted_rating", ascending=[True, False])

# แสดงผลลัพธ์รายชื่อหนังสือที่พยากรณ์ไว้
final_recommendations.show(truncate=False)

# หยุดการทำงานของ Spark
spark.stop()
```

---

**อธิบายการทำงานของโค้ดอย่างละเอียด :**

1. **การเริ่มต้นระบบ (SparkSession):**
    - สร้าง `SparkSession` เพื่อเตรียมสภาพแวดล้อมสำหรับการประมวลผลข้อมูลขนาดใหญ่แบบ Distributed
    - กำหนด `spark.driver.memory` เป็น 4g เพื่อให้รองรับการคำนวณผ่าน RAM ได้อย่างลื่นไหล
2. **การโหลดและล้างข้อมูล (Data Loading & Cleaning):**
    - ใช้ `spark.read.csv` เพื่อโหลดข้อมูล `ratings.csv` และ `books.csv`
    - ใช้ `.dropna()` เพื่อกำจัดแถวที่ข้อมูลไม่ครบถ้วน ป้องกันความคลาดเคลื่อนในการคำนวณ
3. **การสร้างโมเดลด้วย ALS (Alternating Least Squares):**
    - ใช้อัลกอริทึม **ALS** ซึ่งเป็นเทคนิค **Collaborative Filtering** ที่มีประสิทธิภาพสูงในการหาความสัมพันธ์ระหว่างผู้ใช้ (User) และหนังสือ (Book)
    - ตั้งค่า `maxIter=10` (รอบการเรียนรู้) และ `regParam=0.1` (ค่าป้องกันการ Overfitting)
4. **การวัดผล (Evaluation):**
    - แบ่งข้อมูลเป็น **Train (80%)** และ **Test (20%)**
    - ใช้ชุดข้อมูล Test มาทดสอบเพื่อคำนวณค่า **RMSE (Root Mean Square Error)** เพื่อดูว่าผลการทำนายคลาดเคลื่อนจากความชอบจริงของมนุษย์กี่ระดับดาว
5. **การแสดงผลคำแนะนำ (Recommendations):**
    - ใช้ฟังก์ชัน `explode` เพื่อกระจายข้อมูลจาก Array ของคำแนะนำออกมาเป็นแถวรายการหนังสือ
    - ใช้การ `join` ตารางกับข้อมูล `books.csv` เพื่อดึงชื่อหนังสือและชื่อผู้แต่งออกมาแสดงผลให้ผู้ใช้อ่านเข้าใจได้ทันที

---

**ผลลัพธ์ที่ได้ :**

จากการรันระบบโปรแกรม PySpark สามารถจัดการประมวลผล Big Data จำนวนกว่า 6 ล้านรายการได้ออกมาอย่างมีประสิทธิภาพ

1. **ด้านความแม่นยำ:**
   โมเดลมีค่าความคลาดเคลื่อน (RMSE) อยู่ที่ประมาณ **0.8195**
   ความหมายคือ: ค่าความคลาดเคลื่อนต่ำ ซึ่งแปลว่าเมื่อระบบทดสอบทายดาวให้หนังเรื่องหนึ่ง มันจะทายดาวคลาดเคลื่อนจากความเป็นจริงเฉลี่ยไม่ถึง 1 ดาว (อยู่ในเกณฑ์ที่รับได้และทำนายความชอบได้แม่นยำสำหรับระบบให้คะแนนเต็ม 5 ดาว)
2. **ด้านการทำคำแนะนำหน้าแอปพลิเคชันให้ผู้ใช้ทำงานจริง (ตัวอย่างหน้า Terminal):**
   ระบบสามารถคัดเลือกหนังสือและทำนายคะแนนความชอบจัดระเบียบ 5 อันดับแรกให้กับผู้ใช้งานแต่ละคนได้อย่างชัดเจน ตัวอย่างเช่น Output จริงที่ปรากฎบนหน้าจอ Terminal ระบบได้ทำงานตามขั้นตอนและจัดเรียงอันดับผลลัพธ์ให้ดังนี้

> 📷 **[ให้คุณแคปภาพหน้าจอ Terminal ที่รันเสร็จแล้ว ที่เป็นตารางและข้อความแบบด้านล่างนี้ มาแปะแทรกไว้ตรงนี้ได้เลยครับ]**

```text
============================================================
[1] กำลังเริ่มต้น Spark Session...
============================================================
WARNING: Using incubator modules: jdk.incubator.vector
Using Spark's default log4j profile: org/apache/spark/log4j2-defaults.properties
Setting default log level to "WARN".
To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
26/03/09 02:24:30 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable

============================================================
[2] กำลังโหลดข้อมูล Books และ Ratings...
============================================================
--- ตัวอย่างข้อมูล ratings ---
+-------+-------+------+
|user_id|book_id|rating|
+-------+-------+------+
|      1|    258|     5|
|      2|   4081|     4|
|      2|    260|     5|
+-------+-------+------+
only showing top 3 rows

============================================================
[3] เตรียมข้อมูลและแบ่ง Train / Test...
============================================================
-> แบ่งข้อมูลเสร็จสิ้น: Train 80%, Test 20%

============================================================
[4] กำลัง Train Model Recommendation (ALS)... อาจใช้เวลาสักครู่...
============================================================
-> Train Model สำเร็จ!

============================================================
[5] กำลังประเมินผลความแม่นยำด้วย RMSE...
============================================================
✅ Root Mean Square Error (RMSE) ของระบบแนะนำ = 0.8195

============================================================
[6] แสดงผลการแนะนำหนังสือให้ 5 ผู้ใช้ (Top 5 Recommendations)...
============================================================
+-------+---------------------------------------------------------------------------------+------------------------------------------+----------------+
|user_id|title                                                                            |authors                                   |predicted_rating|
+-------+---------------------------------------------------------------------------------+------------------------------------------+----------------+
|463    |The Complete Calvin and Hobbes                                                   |Bill Watterson                            |4.034318        |
|463    |This is Not My Hat                                                               |Jon Klassen                               |4.006405        |
|463    |The Authoritative Calvin and Hobbes: A Calvin and Hobbes Treasury                |Bill Watterson                            |3.9743338       |
|463    |ESV Study Bible                                                                  |Anonymous, Lane T. Dennis, Wayne A. Grudem|3.967501        |
|463    |The Divan                                                                        |Hafez                                     |3.964475        |
|1238   |Kindle Paperwhite User's Guide                                                   |Amazon                                    |4.6414757       |
|1238   |Bumi Manusia                                                                     |Pramoedya Ananta Toer                     |4.3772182       |
|1238   |The Gates of Rome (Emperor, #1)                                                  |Conn Iggulden                             |4.3564744       |
|1238   |The Nightingale                                                                  |Kristin Hannah                            |4.3502254       |
|1238   |Love Wins: A Book About Heaven, Hell, and the Fate of Every Person Who Ever Lived|Rob Bell                                  |4.34654         |
|1580   |The Divan                                                                        |Hafez                                     |4.855482        |
|1580   |Humans of New York: Stories                                                      |Brandon Stanton                           |4.715762        |
|1580   |The Complete Calvin and Hobbes                                                   |Bill Watterson                            |4.707438        |
|1580   |Just Mercy: A Story of Justice and Redemption                                    |Bryan Stevenson                           |4.6567717       |
|1580   |Attack of the Deranged Mutant Killer Monster Snow Goons                          |Bill Watterson                            |4.6069727       |
|1645   |Mark of the Lion Trilogy                                                         |Francine Rivers                           |4.4151607       |
|1645   |The Day the Crayons Quit                                                         |Drew Daywalt, Oliver Jeffers              |4.308016        |
|1645   |The Nightingale                                                                  |Kristin Hannah                            |4.306144        |
|1645   |Humans of New York: Stories                                                      |Brandon Stanton                           |4.2991996       |
|1645   |All Souls: A Family Story from Southie                                           |Michael Patrick MacDonald                 |4.2947087       |
|2366   |One Piece, Volume 38: Rocketman!! (One Piece, #38)                               |Eiichirō Oda                              |5.3370814       |
|2366   |Words of Radiance (The Stormlight Archive, #2)                                   |Brandon Sanderson                         |5.324891        |
|2366   |The Indispensable Calvin and Hobbes                                              |Bill Watterson                            |5.3074975       |
|2366   |The Wise Man's Fear (The Kingkiller Chronicle, #2)                               |Patrick Rothfuss                          |5.2967544       |
|2366   |Hellsing, Vol. 01 (Hellsing, #1)                                                 |Kohta Hirano, Duane Johnson               |5.2849355       |
+-------+---------------------------------------------------------------------------------+------------------------------------------+----------------+

🎉 รันเสร็จสมบูรณ์! หยุดการทำงานของ Spark...
SUCCESS: The process with PID 18776 (child process of PID 15892) has been terminated.
SUCCESS: The process with PID 15892 (child process of PID 14040) has been terminated.
SUCCESS: The process with PID 14040 (child process of PID 2004) has been terminated.
(.venv) PS C:\Users\USER\OneDrive\Documents\BEAM\ANN\COE64-345 Big Data Implementation\Project 2> 
```

**คำอธิบายผลลัพธ์ทีละขั้นตอน (Step-by-Step Explanation):**

- **ขั้นตอน [1]-[2] :** โปรแกรมเริ่มต้นกระบวนการด้วยการสร้าง Spark Session และนำเข้าชุดข้อมูลสำเร็จ โดยแสดงตัวอย่างข้อมูลดิบออกมาให้ดู 3 บรรทัดแรก เพื่อยืนยันว่าเข้าถึงไฟล์ `.csv` ได้จริง (ประกอบด้วย `user_id`, `book_id`, และ `rating`)
- **ขั้นตอน [3]-[4] :** โปรแกรมทำการแบ่งกลุ่มข้อมูล 80/20 และป้อนให้ Machine Learning อัลกอริทึม ALS เริ่มทำการเรียนรู้และแกะแพทเทิร์นพฤติกรรมผู้ใช้
- **ขั้นตอน [5] :** โมเดลถูกเทรนเสร็จสิ้น และทำแบบทดสอบให้คะแนนความแม่นยำตัวเองด้วยมาตรวัด **RMSE ออกมาที่ 0.8195** ซึ่งถือว่าเป็นค่าความคลาดเคลื่อนที่อยู่ในระดับที่ต่ำ ทายดาวคลาดเคลื่อนไม่ถึง 1 ดาว (ยิ่งตัวเลขน้อย ยิ่งมีประวิทธิภาพดี)
- **ขั้นตอน [6] :** โปรแกรมสุ่มหยิบผู้ใช้ออกมา 5 ตัวอย่าง (UserID: 463, 1238, 1580, 1645, 2366) และสั่งให้โมเดลประมวณผลลัพธ์หนังสือที่ผู้ใช้แต่ละคนน่าจะให้คะแนนเยอะที่สุด (ทายใจไล่จากมากลงน้อย 5 อันดับแรก) โดยผลจับคู่ชื่อหนังสือ (`title`) และชื่อผู้แต่ง (`authors`) อย่างเป็นระเบียบตามตาราง ซึ่งผลที่ออกมามีความสมเหตุสมผลและเหมาะสมกับ User คนนั้นๆ

**ประโยชน์ของระบบนี้ และการนำไปใช้ :**

1. **การปรับปรุงประสบการณ์ผู้ใช้ (Personalization):** ระบบสามารถนำเสนอหนังสือที่ตรงใจผู้ใช้แต่ละคนได้โดยตรง ช่วยลดเวลาในการค้นหาหนังสือเล่มใหม่ และสร้างความประทับใจให้กับผู้ใช้
2. **การเพิ่มยอดขายและ Engagement:** ในเชิงธุรกิจ E-commerce (เช่น ร้านหนังสือออนไลน์) ระบบแนะนำที่มีประสิทธิภาพจะช่วยกระตุ้นยอดขาย (Cross-selling/Up-selling) และทำให้ผู้ใช้กลับมาใช้งานแพลตฟอร์มบ่อยขึ้น
3. **การจัดการข้อมูลขนาดใหญ่ (Scalability):** เนื่องจากการใช้ PySpark ระบบนี้จึงรองรับการขยายตัวเพื่อจัดการกับข้อมูลผู้ใช้และหนังสือจำนวนมหาศาลในอนาคตได้อย่างเสถียร
4. **ความแม่นยำในการวิเคราะห์พฤติกรรม:** อัลกอริทึม ALS ช่วยให้เราเข้าใจความสัมพันธ์ที่ซับซ้อนระหว่างรสนิยมความชอบที่ไม่สามารถมองเห็นได้ด้วยตาเปล่า ช่วยให้ธุรกิจวางแผนกลยุทธ์การตลาดได้แม่นยำยิ่งขึ้น

**แนวทางการนำไปใช้งานจริง (Implementation Roadmap) :**

1. **การเชื่อมต่อกับหน้าร้าน (Backend Integration):** นำผลลัพธ์จากโมเดล ALS ไปเก็บไว้ใน Database (เช่น Redis หรือ MongoDB) เพื่อให้ระบบ API ของแอปพลิเคชันดึงไปแสดงผลที่หน้า "Recommended for You" ได้ในระดับมิลลิวินาที
2. **การทำงานแบบ Real-time (Streaming):** พัฒนาต่อยอดโดยใช้ **Spark Streaming** เพื่อรับข้อมูลการคลิกหรือการซื้อแบบทันที แล้วนำมาอัปเดตคำแนะนำให้เป็นปัจจุบัน (Near Real-time Recommendation)
3. **ระบบ Hybrid Recommendation:** ผสมผสาน ALS กับเทคนิค Content-based (เช่น แนะนำหนังสือจากหมวดหมู่ที่ผู้ใช้ชอบ) เพื่อแก้ปัญหา **Cold Start** สำหรับผู้ใช้ใหม่ที่ยังไม่มีประวัติการให้คะแนน
4. **การทำ A/B Testing:** นำระบบแนะนำไปทดสอบเปรียบเทียบกับระบบเดิม เพื่อวัดผลอัตราการคลิก (CTR) และยอดขายที่เพิ่มขึ้นจริง เพื่อนำข้อมูลมาปรับปรุงโมเดลให้ดียิ่งขึ้น

---

**สรุปผลรวม:** โปรเจกต์นี้แสดงให้เห็นถึงศักยภาพของการสร้าง Recommendation System นำข้อมูลการให้คะแนนของลูกค้าที่มีปริมาณมหาศาล (Big Data) มาร่วมกันหารูปแบบพฤติกรรมผ่านโมเดลเทคโนโลยี Machine Learning ได้อย่างเสถียร รวมถึงได้ผลลัพธ์แบบรายบุคคล (Personalized recommendation) ที่สามารถนำไปประกอบใช้งานกับระบบ E-Commerce ร้านขายหนังสือหน้าเว็บต่อได้ทันที
