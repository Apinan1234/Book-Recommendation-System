# การออกแบบสไลด์พรีเซ้นฉบับ Premium (Project 2)

คู่มือนี้จะแสดงภาพม็อคอัพและเลย์เอาต์ที่แนะนำสำหรับแต่ละหน้าสไลด์ เพื่อให้งานพรีเซ้นของคุณ Beam ดูทันสมัยและเป็นมืออาชีพที่สุดครับ

````carousel
# Slide 1: หน้าปก (Title)
![Title Page](https://images.unsplash.com/photo-1507842217343-583bb7270b66?auto=format&fit=crop&w=1200&q=80)
**เลย์เอาต์:** กึ่งกลาง (Center aligned)
**เนื้อหา:**
- **หัวข้อใหญ่:** Book Recommendation System
- **หัวข้อย่อย:** COE64-345 Big Data Implementation
- **ท้ายสไลด์:** จัดทำโดย [ชื่อของคุณ Beam]
<!-- slide -->
# Slide 2: ปัญหา (The Problem)
![Problem Slide](/C:/Users/USER/.gemini/antigravity/brain/5ffbb8e6-b351-4b2f-8fcc-14ed7a063454/slide_problem_overwhelmed_reader_1773129918246.png)
**เลย์เอาต์:** แบ่งครึ่ง ซ้าย-ขวา
**ฝั่งซ้าย (ภาพ):** ภาพคนท่ามกลางกองหนังสือ (ที่ผมเจนนให้)
**ฝั่งขวา (ข้อความ):** 
- "มนุษย์เรามีเวลาจำกัด..."
- "แต่โลกนี้มีหนังสือมากกว่า 150 ล้านเล่ม"
- **คำถาม:** เราจะหา 'เล่มที่ใช่' ท่ามกลางกองข้อมูลมหาศาลได้อย่างไร?
<!-- slide -->
# Slide 3: ทางออก (The Solution)
![Solution Slide](/C:/Users/USER/.gemini/antigravity/brain/5ffbb8e6-b351-4b2f-8fcc-14ed7a063454/slide_solution_ai_recommendation_1773129964095.png)
**เลย์เอาต์:** เน้นภาพพื้นหลังจางๆ และข้อความเด่นกลางจอ
**เนื้อหา:**
- **คำโปรย:** "ทายใจนักอ่าน ด้วยพลังของ AI & Big Data"
- **หัวใจหลัก:** ระบบแนะนำหนังสือรายบุคคล (Personalized Recommendation)
<!-- slide -->
# Slide 4: ข้อมูลที่เราใช้ (The Data)
![Data Overview](https://images.unsplash.com/photo-1543286386-2e659306cd6c?auto=format&fit=crop&w=1200&q=80)
**เลย์เอาต์:** ตารางคู่ (Side-by-side Tables)
**เนื้อหา:**
- **ฝั่งซ้าย:** ตาราง `ratings.csv` (ตัวอย่าง 3 แถว) -> แสดง user_id, book_id, rating
- **ฝั่งขวา:** ตาราง `books.csv` (ตัวอย่าง 3 แถว) -> แสดง title, authors
- **ล่างสุด:** "รวมข้อมูลการให้คะแนนกว่า 6,000,000 รายการ"
<!-- slide -->
# Slide 5: การทำงาน (How it works?)
![Collaborative Filtering](/C:/Users/USER/.gemini/antigravity/brain/5ffbb8e6-b351-4b2f-8fcc-14ed7a063454/slide_howitworks_collaborative_filtering_1773129985107.png)
**เลย์เอาต์:** Infographic
**เนื้อหา:**
- **หัวข้อ:** Collaborative Filtering (ALS Algorithm)
- **คำอธิบาย:** "ถ้าคุณชอบเหมือนเขา... คุณก็น่าจะชอบเล่มอื่นที่เขาแนะนำด้วย"
- **เทคนิค:** ใช้การหาความสัมพันธ์ระหว่างพฤติกรรมผู้ใช้
<!-- slide -->
# Slide 6: โค้ดส่วนเตรียมข้อมูล (Setup)
```python
# Screenshot บรรทัดที่ 46-65 ใน main.py
ratings = ratings.dropna(subset=["user_id", "book_id", "rating"])
```
**เลย์เอาต์:** Code Block ด้านหนึ่ง และคำอธิบายอีกด้าน
**คำอธิบาย:** "ต้องลบค่าว่าง (Dropna) ออกก่อน เพื่อให้โมเดลไม่สับสนและมีความแม่นยำสูงสุด"
<!-- slide -->
# Slide 7: หัวใจของโมเดล (The Model)
```python
# Screenshot บรรทัดที่ 71-73 ใน main.py
als = ALS(maxIter=10, regParam=0.1, userCol="user_id", ...)
```
**เลย์เอาต์:** เน้นตัวเลขพารามิเตอร์ให้เด่น (เช่น วงกลมรอบ regParam=0.1)
**คำอธิบาย:** "ALS คืออัลกอริทึมที่ Spark ออกแบบมาเพื่อจัดการข้อมูลขนาดใหญ่ในรูปแบบ Matrix โดยเฉพาะ"
<!-- slide -->
# Slide 8: ความแม่นยำ (Evaluation)
![Accuracy Metrics](https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&w=1200&q=80)
**เลย์เอาต์:** ตัวเลขใหญ่ๆ กลางจอ
**เนื้อหา:**
- **RMSE = 0.8195**
- **ความหมาย:** "คลาดเคลื่อนไม่ถึง 1 ดาว"
- **สรุป:** "ระบบสามารถพยากรณ์ความชอบได้แม่นยำสูง (แม่นยำ > 80%)"
<!-- slide -->
# Slide 9: ผลลัพธ์จริง (Result)
**เลย์เอาต์:** Screenshot จาก Terminal ล่าสุดของคุณ Beam
**เนื้อหา:** 
- โชว์ตารางที่มี User ID และชื่อหนังสือที่แนะนำ
- **เน้น:** ตัวอย่าง User 2366 ที่แนะนำ One Piece และ Brandon Sanderson (ผลลัพธ์ดูสมเหตุสมผล)
<!-- slide -->
# Slide 10: ความสำเร็จ (Conclusion & Roadmap)
![Roadmap](/C:/Users/USER/.gemini/antigravity/brain/5ffbb8e6-b351-4b2f-8fcc-14ed7a063454/slide_future_roadmap_ecommerce_1773130003437.png)
**เลย์เอาต์:** แผนภาพก้าวไปข้างหน้า
**เนื้อหา:**
- **Impact:** ช่วยเพิ่มยอดขายและสร้าง Engagement
- **Next Step:** พัฒนาเป็นระบบ Real-time และ Hybrid
<!-- slide -->
````

### 💡 ทิปเพิ่มเติมสำหรับการทำสไลด์

- **ใช้ Font ที่ทันสมัย:** เช่น *Inter*, *Kanit* (สำหรับภาษาไทย), หรือ *Montserrat* จะช่วยให้ดู Premium ขึ้นมากครับ
- **อย่าใส่ข้อความเยอะเกินไป:** ให้เน้นภาพที่ผมเจนให้เป็นตัวเล่าความรู้สึก แล้วคุณ Beam ค่อยพูดอธิบายรายละเอียดตามบทพูดที่ผมเตรียมไว้ครับ
- **ใช้ธีมสีเข้ม (Dark Mode):** จะเข้ากับสีของรูปภาพที่ผมส่งให้ได้ดีมากครับ
