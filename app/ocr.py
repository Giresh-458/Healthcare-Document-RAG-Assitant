import pytesseract
from PIL import Image
import io
from app.config import TESSERACT_CMD

# Configure Tesseract path for Windows
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

def extract_text_from_image(image_bytes: bytes) -> str:
    """
    Extracts text from an image using Tesseract OCR.
    
    Args:
        image_bytes (bytes): The raw image data (e.g., PNG or JPEG).
        
    Returns:
        str: The extracted text, or empty string if OCR fails.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        print(f"OCR Error: {e}")
        return ""
