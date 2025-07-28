import pytesseract
from PIL import Image

# Open the image
img = Image.open('/image_to_text/img.png')
# Perform text recognition
text = pytesseract.image_to_string(img)
print(text)