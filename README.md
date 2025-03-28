# 📝 OCR-Based Text Recognition Project

Welcome to the **OCR-Based Text Recognition Project for John Deere!** 🖥️📸
This project utilizes **EasyOCR, OpenCV, and PyTorch** to recognize text from images in real-time.

---

## 🛠 Installation
To install and set up the project, follow these steps:

### 1️⃣ Clone the Repository:
```
git clone https://github.com/FazonPlay/the8siders
cd OCR_Project
```

### 2️⃣ Install Dependencies:
Make sure you have **Python** installed, then run:
```
pip install opencv-python   
pip install numpy<2.0.0  # Must be below 2.0 to avoid compatibility issues  
pip install pillow
pip install easyocr
pip install torch==2.6.0 torchvision torchaudio
pip install "PyQt5>=5.15.2"
```

### 3️⃣ Install Tesseract OCR:
Download and install Tesseract from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki).

After installation, update the path in the code:
```
pytesseract.pytesseract.tesseract_cmd = r'C:\PATH_TO_FILE\tesseract.exe'  # Adjust this based on your system
```

### 4️⃣ Configure Environment Variables (Optional but Recommended):
Add Tesseract to your system’s PATH variable to avoid manual path updates in the code.

---

## 💻 System Requirements
To ensure smooth operation, your system should meet the following requirements:

✅ **CPU:** A powerful processor for real-time processing  
✅ **RAM:** At least 16GB for optimal performance  
✅ **GPU (Optional):** A dedicated GPU can improve performance but is not required  
✅ **Camera:** A high-resolution camera for better OCR accuracy  

---

## 🔥 Important Notes
- Avoid installing the **CUDA version of PyTorch** as it may cause compatibility issues with the gpu.
- **NumPy version must be below 2.0** to prevent conflicts with OCR libraries.
- Keep **torch==2.6.0**, as this version has been tested and works properly.

---

## 🤝 Contributors
- **Nikita**
- **David**
- **Julien**
- **Mourad**

---
✨ *Thank you for checking out our project!* ✨

