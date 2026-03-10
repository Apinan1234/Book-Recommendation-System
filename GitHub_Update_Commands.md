# 🚀 วิธีอัปเดตข้อมูลขึ้น GitHub (สำหรับรันเอง)

คุณ Beam สามารถก๊อปปี้คำสั่งด้านล่างนี้ไปวางใน Terminal (PowerShell) เพื่ออัปเดตงานล่าสุดได้เลยครับ **ไม่จำเป็นต้องลบของเก่าครับ** Git จะทำการเปรียบเทียบและอัปเดตเฉพาะส่วนที่เปลี่ยนแปลงให้เอง

### ขั้นตอนการรัน

1. **ไปที่โฟลเดอร์โปรเจกต์:**

    ```powershell
    cd "c:\Users\USER\OneDrive\Documents\BEAM\ANN\COE64-345 Big Data Implementation\Book Recommendation System"
    ```

2. **เตรียมไฟล์ทั้งหมดที่จะอัปโหลด:**

    ```powershell
    git add .
    ```

3. **บันทึกการเปลี่ยนแปลง (Commit):**

    ```powershell
    git commit -m "Update: Final report, presentation guide, and NotebookLM prompt"
    ```

4. **ส่งข้อมูลขึ้น GitHub:**

    ```powershell
    git push origin main
    ```

---

### ❓ คำถามที่พบบ่อย

* **ต้องลบของเก่าไหม?**
  * **ไม่ต้องครับ** Git ออกแบบมาเพื่อ "ต่อยอด" จากของเดิม การรัน `git push` จะเป็นการส่งเฉพาะไฟล์ที่เพิ่มใหม่หรือถูกแก้ไขขึ้นไปบน GitHub ให้เองครับ
* **ถ้าติด Error เรื่อง `rejected` หรือ `fetch first`?**
  * ถ้าคุณมั่นใจว่าไฟล์ในเครื่องล่าสุดที่สุดแล้ว ให้พุชด้วยคำสั่งแรง (force):
        `git push origin main --force`
