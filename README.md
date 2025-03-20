# AT LAST, THERE IS A GODDAMN README FILE

- As of the 13th commit on 20/03 you also need to install PyQt5:
  Windows:
    pip install "PyQt5>=5.15.2"

  Linux:
      pip install "PyQt5>=5.15.2"
  
    Install additional system packages:
      sudo apt-get install python3-pyqt5
      sudo apt-get install python3-pyqt5.qtwebengine

  Mac:
   1. Install PyQt5 using Homebrew:
   ```bash
     brew install pyqt@5


- As of the 9th commit on the 16/03 here are the dependencies needed:

 
### Core dependencies
pip install opencv-python   
pip install numpy<2.0.0  # Must be below 2.0 to avoid compatibility issues  
pip install pillow

### OCR engines
pip install easyocr  
pip install pytesseract   
pip install paddleocr 

### PyTorch 
pip install torch==2.6.0 torchvision torchaudio


# System Requirements
### Tesseract OCR Engine  

Download and install from GitHub https://github.com/UB-Mannheim/tesseract/wiki  
Update path in code:  
pytesseract.pytesseract.tesseract_cmd = r'C:\PATH_TO_FILE\tesseract.exe' (it varies per system)

### Camera  
Webcam or USB camera connected to system    
If you want to use the laptop camera (or the default camera) you have to apply the index=0
```
    def __init__(self, camera_index=0):
```
replace the 0 with 1 if you want an external camera (or wireless phone camera)

# Notes 
Avoid installing the CUDA version of PyTorch that caused the issues
The NumPy version must be below 2.0 to avoid compatibility errors with the OCR libraries
Keep your working torch==2.6.0 installation that was functioning properly

I suggest y'all read the code as there are a LOT of stuff that have been done since the last commit  

Also, please check the main.py file as you can uncomment the method of scanning you want,
you can either choose to capture 3 images live or you can import an image from the system.



