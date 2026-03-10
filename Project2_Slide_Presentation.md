---
marp: true
theme: default
paginate: true
backgroundColor: #f0f4f8
---

# 📚 ระบบแนะนำหนังสือ (Book Recommendation System)

**Project 2: Big Data Implementation**

นำเสนอโดย: [ชื่อ - นามสกุล]
อ้างอิงชุดข้อมูล: Goodbooks-10k dataset

---

# 🎯 1. วัตถุประสงค์ของโปรเจ็กต์ (Objective)

เราจะทำการ **Recommend อะไร?**

- **เป้าหมาย:** พัฒนาระบบแนะนำหนังสือ (Personalized Book Recommendation)
- **สิ่งที่จะแนะนำ:** "รายชื่อหนังสือ 5 เล่ม" ที่คาดว่าผู้ใช้งานแต่ละคนน่าจะชื่นชอบ
- **แนวคิด:** ใช้พฤติกรรมการให้คะแนน (Rating) ในอดีต เพื่อหาจุดร่วมความชอบของผู้ใช้ (Collaborative Filtering)

---

# 📊 2. ข้อมูลที่นำมาใช้ (Dataset)

**Goodbooks-10k** (จาก GitHub)
ประกอบด้วยหนังสือยอดนิยม 10,000 เล่ม และรีวิว 6 ล้านรายการ

**ตารางที่ 1: `ratings.csv` (ตารางพฤติกรรม)**

- `user_id`: รหัสประจำตัวของผู้ใช้
- `book_id`: รหัสประจำตัวของหนังสือ
- `rating`: คะแนนที่ผู้ใช้ให้กับหนังสือเล่มนั้น ๆ (1-5 ดาว)

**ตารางที่ 2: `books.csv` (ตารางรายละเอียด)**

- `book_id`: รหัสประจำตัวของหนังสือ
- `title`: ชื่อหนังสือ
- `authors`: ชื่อผู้แต่งหนังสือ
- `average_rating`: คะแนนรีวิวเฉลี่ย

---

# ⚙️ 3. การเตรียมความพร้อม (Setup & Installation)

ก่อนเริ่มทำงาน จำเป็นต้องติดตั้งเครื่องมือต่อไปนี้ผ่าน Terminal:

**1. ติดตั้ง PySpark:** สำหรับจัดการและประมวลผล Big Data

```bash
pip install pyspark
```

**2. ติดตั้ง NumPy:** สำหรับการคำนวณคณิตศาสตร์และ Matrix เบื้องหลัง

```bash
pip install numpy
```

---

# 🛠️ 4. ขั้นตอนการพัฒนาระบบ (Methodology)

การประมวลผล Big Data ด้วย **PySpark** แบ่งเป็น 5 ขั้นตอน:

1. **Data Loading:** โหลดข้อมูล `ratings.csv` และ `books.csv`
2. **Train/Test Split:** แบ่งข้อมูล Train (80%) เอาไว้สอน AI ส่วน Test (20%) เก็บไว้เป็นข้อสอบวัดความแม่นยำ
3. **Build Model:** ใช้เทคนิค **Collaborative Filtering** ด้วย **ALS (Alternating Least Squares)** เพื่อหารสนิยมแฝงที่คนกลุ่มเดียวกันมีคล้ายกัน
4. **Evaluation:** ประเมินความแม่นยำคะแนนด้วยค่า RMSE (ประเมินว่าทายดาวคลาดเคลื่อนไปกี่ดวง)
5. **Recommendation:** จัดอันดับและแนะนำหนังสือ Top 5 ที่ตรงพฤติกรรมผู้ใช้ที่สุด

---

# 💻 5. โค้ดการทำงาน (PySpark Implementation)

ตัวอย่างการสร้างโมเดล ALS:

```python
from pyspark.ml.recommendation import ALS

# กำหนด Config ให้โมเดล
als = ALS(
    maxIter=10,               # รันประมวลผล 10 รอบ เพื่อหาจุดที่สมดุลที่สุด
    regParam=0.1,             # ป้องกันไม่ให้ AI จำคำตอบมากไป (Overfitting)
    userCol="user_id",        # ระบุคอลัมน์รหัสผู้ใช้งาน
    itemCol="book_id",        # ระบุคอลัมน์รหัสหนังสือ (รายการที่แนะนำ)
    ratingCol="rating",       # ระบุคอลัมน์คะแนนรีวิว
    coldStartStrategy="drop"  # ป้องกันโปรแกรมพัง หากเจอ User ใหม่แกะกล่องตอน Test
)

# เริ่มทำการ Train Model ด้วยชุดข้อมูล 80%
model = als.fit(trainingData)
```

---

# 📈 6. การวัดผลความแม่นยำ (Evaluation)

ตัวอย่างโค้ดและผลประเมิน:

```python
from pyspark.ml.evaluation import RegressionEvaluator

predictions = model.transform(testData)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
```

**📌 ผลลัพธ์ความแม่นยำ:**

- **RMSE (Root Mean Square Error) ≈ 0.8x**
- หมายความว่าระบบทำนายดาวคลาดเคลื่อนเฉลี่ยไม่ถึง 1 ดาว (อยู่ในเกณฑ์ที่รับได้และนำไปใช้งานได้จริง)

---

# ✨ 7. ผลลัพธ์การแนะนำหนังสือ (Result)

ตัวอย่างผลลัพธ์หนังสือ Top 5 ที่ระบบแนะนำให้ **User rหัส 1**:

| ลำดับแนะนำ | ชื่อหนังสือ (Title) | ผู้แต่ง (Authors) | คะแนนคาดการณ์ |
|:---:|---|---|:---:|
| 1 | The Calvin and Hobbes Tenth Anniversary Book | Bill Watterson | 5.0 |
| 2 | The Complete Calvin and Hobbes | Bill Watterson | 4.9 |
| 3 | Harry Potter Boxset | J.K. Rowling | 4.8 |
| 4 | It's a Magical World (Calvin and Hobbes) | Bill Watterson | 4.8 |
| 5 | The Lorax | Dr. Seuss | 4.7 |

*(ระบบสามารถจับคู่หนังสือสไตล์ครอบครัว/แฟนตาซีให้ผู้ใช้รายนี้ได้อย่างแม่นยำ)*

---

# ✅ 8. สรุปผล (Conclusion)

- ระบบ **Book Recommendation System** สามารถทำงานบน Framework ของ PySpark เพื่อประมวลผลระดับ Big Data (6 ล้านเรคคอร์ด) ได้อย่างรวดเร็ว
- อัลกอริทึม **ALS (Collaborative Filtering)** สามารถดึงพฤติกรรมความชอบแฝงของผู้ใช้ออกมาได้ดีเยี่ยม
- ระบบสามารถให้ผลลัพธ์แบบรายบุคคล (Personalized) นำข้อมูลไปต่อยอดทำฟังก์ชัน "Recommended for you" บนแพลตฟอร์ม E-Commerce หรือแอปอ่านหนังสือได้จริง

---

# 🙏 Q&A

**ขอบคุณที่รับฟัง**
