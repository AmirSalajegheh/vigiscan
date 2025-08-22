import tkinter as tk
from tkinter import ttk
import subprocess
import json

# تابع برای اجرا
def run_detection():
    selected_options = {
        'helmet': helmet_var.get(),
        'gloves': gloves_var.get(),
        'glasses': glasses_var.get(),
        'vest': vest_var.get(),
        'suit': suit_var.get()
    }
    
    # ذخیره تنظیمات به فایل config
    with open("config.json", "w") as f:
        json.dump(selected_options, f)

    # اجرای فایل پردازش اصلی
    subprocess.Popen(["python", "detection.py"])

# پنجره اصلی
root = tk.Tk()
root.title("PPE Detection Configuration")
root.geometry("300x300")

# برچسب عنوان
ttk.Label(root, text="موارد مورد بررسی:", font=("B Titr", 14)).pack(pady=10)

# متغیرها
helmet_var = tk.BooleanVar(value=True)
gloves_var = tk.BooleanVar(value=True)
glasses_var = tk.BooleanVar(value=True)
vest_var = tk.BooleanVar(value=True)
suit_var = tk.BooleanVar(value=True)

# چک‌باکس‌ها
ttk.Checkbutton(root, text="کلاه ایمنی", variable=helmet_var).pack(anchor="w", padx=20)
ttk.Checkbutton(root, text="دستکش", variable=gloves_var).pack(anchor="w", padx=20)
ttk.Checkbutton(root, text="عینک", variable=glasses_var).pack(anchor="w", padx=20)
ttk.Checkbutton(root, text="جلیقه", variable=vest_var).pack(anchor="w", padx=20)
ttk.Checkbutton(root, text="لباس ایمنی", variable=suit_var).pack(anchor="w", padx=20)

# دکمه اجرا
ttk.Button(root, text="شروع", command=run_detection).pack(pady=20)

root.mainloop()