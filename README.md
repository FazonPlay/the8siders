### Dependencies for the project

# Works for all OS

```
pip install opencv-python   
pip install numpy<2.0.0  # Must be below 2.0 to avoid compatibility issues  
pip install pillow
pip install easyocr
pip install torch==2.6.0 torchvision torchaudio
pip install "PyQt5>=5.15.2"
```

# System Requirements
Powerful CPU for real-time processing 

Download and install from GitHub https://github.com/UB-Mannheim/tesseract/wiki  
Update path in code:  
pytesseract.pytesseract.tesseract_cmd = r'C:\PATH_TO_FILE\tesseract.exe' (it varies per system)

# Notes 
Avoid installing the CUDA version of PyTorch that caused the issues
The NumPy version must be below 2.0 to avoid compatibility errors with the OCR libraries
Keep the torch==2.6.0 as that was working properly
