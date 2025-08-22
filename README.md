# 🦺 VigiScan  

**VigiScan** یک سیستم هوشمند **تشخیص تجهیزات ایمنی (PPE)** بر پایه‌ی **YOLOv9** است.  
این سیستم بررسی می‌کند که آیا کارگران از تجهیزات ایمنی ضروری استفاده کرده‌اند یا نه، از جمله:  
- 🪖 کلاه ایمنی (Helmet)  
- 🧤 دستکش (Gloves)  
- 👓 عینک ایمنی (Safety Glasses)  
- 👕 جلیقه ایمنی (Safety Vest)  
- 🧥 لباس ایمنی (Safety Suit)  

این پروژه دارای یک **داشبورد گرافیکی (Tkinter UI)** است که تنظیمات و نمایش نتایج را بسیار ساده می‌کند.  

---

## 🚀 امکانات  

✅ تشخیص بلادرنگ با YOLOv9  
✅ داشبورد قابل تنظیم (فعال/غیرفعال کردن هر PPE)  
✅ پشتیبانی از **وبکم** و **فایل ویدئویی**  
✅ ذخیره ویدئوی خروجی با باکس‌ها و برچسب‌ها  
✅ رابط کاربری ساده با **Tkinter**  

---

## 📂 ساختار پروژه  

```
PPE/
│   main.py            # اجرای اصلی همراه با رابط گرافیکی
│   detection.py       # منطق تشخیص با YOLOv9
│   config.json        # فایل پیکربندی (PPE و تنظیمات)
│   requirements.txt   # کتابخانه‌های موردنیاز
│
└───Data/
    │   test_input.mp4   # ویدئوی تست
    │   output.mp4       # خروجی پردازش‌شده
    │   yolo9e.pt        # وزن‌های مدل (با Git LFS)
    │   Titr.ttf         # فونت فارسی برای نمایش
```

---

## ⚙️ نصب  

1. کلون کردن ریپو:  
```bash
git clone https://github.com/AmirSalajegheh/vigiscan.git
cd vigiscan
```

2. ساخت محیط مجازی (اختیاری اما توصیه‌شده):  
```bash
python -m venv venv
venv\Scripts\activate      # ویندوز
source venv/bin/activate   # لینوکس / مک
```

3. نصب وابستگی‌ها:  
```bash
pip install -r requirements.txt
```

4. دریافت وزن‌های مدل (در صورت استفاده از LFS):  
```bash
git lfs install
git lfs pull
```

---

## ▶️ اجرا  

### اجرای رابط گرافیکی:  
```bash
python main.py
```

- انتخاب منبع ورودی (وبکم یا فایل ویدئویی)  
- انتخاب آیتم‌های PPE برای بررسی  
- شروع پردازش و ذخیره خروجی  

### اجرای مستقیم بدون UI:  
```bash
python detection.py --input Data/test_input.mp4 --output Data/output.mp4
```

---

## 📊 نمونه خروجی  

| ورودی | خروجی |
|-------|-------|
| 🎥 ویدئو خام | ✅ ویدئو همراه با باکس و برچسب PPE |

---

## 🛠 تکنولوژی‌ها  

- [Python 3.10+](https://www.python.org/)  
- [YOLOv9](https://github.com/WongKinYiu/yolov9)  
- [OpenCV](https://opencv.org/)  
- [Tkinter](https://docs.python.org/3/library/tkinter.html)  

---

## 📌 نقشه راه  

- [ ] افزودن ارسال هشدار (ایمیل / SMS) در صورت نقض PPE  
- [ ] پشتیبانی از چند دوربین همزمان  
- [ ] داشبورد تحت وب با Streamlit / FastAPI  

---

## 👤 نویسنده  

👨‍💻 توسعه‌دهنده: [**Amir Salajegheh**](https://github.com/AmirSalajegheh)  
