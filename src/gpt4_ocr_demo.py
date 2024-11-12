import base64
from openai import OpenAI
from dotenv import load_dotenv
import os
import logging
import time
from PIL import Image
import io

class OCRClient:
    def __init__(self, model="gpt-4o-mini"):
        self._load_environment()
        self.client = self._initialize_openai_client()
        self.model = model
        self.max_retries = 3  # Set maximum retries for API calls
        self.retry_delay = 2  # Delay between retries in seconds

    def _load_environment(self):
        """Load environment variables and check API key."""
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key not found. Please set OPENAI_API_KEY in the environment variables.")

    def _initialize_openai_client(self):
        """Initialize and return an OpenAI client."""
        return OpenAI(api_key=self.api_key)

    def resize_image(self, image_path, max_size=(300, 100)):
        """Resize image to reduce size for faster processing."""
        with Image.open(image_path) as img:
            img.thumbnail(max_size)
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
        return buffered.getvalue()

    def encode_image_to_base64(self, image_data):
        """Encode image data to base64 string."""
        return base64.b64encode(image_data).decode('utf-8')

    def invoke_gpt4o_ocr(self, encoded_image):
        """Invoke GPT-4o-mini to perform OCR with retries for robustness."""
        for attempt in range(self.max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "请对这张图片进行OCR识别，并输出最准确的验证码，以项目列表格式直接输出识别出的结果，不要输出其他内容。"},
                                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded_image}}
                            ]
                        }
                    ],
                    max_tokens=300,
                )
                result = completion.choices[0].message.content.replace("-", "").strip()
                return result
            except Exception as e:
                logging.warning(f"Attempt {attempt + 1} failed with error: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    logging.error("Max retries reached. OCR process failed.")
                    return None

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    try:
        # Create an OCR client instance
        ocr_client = OCRClient()

        # Set image path and load image
        image_path = 'img/226md.png'

        # Resize and encode the image to base64
        resized_image_data = ocr_client.resize_image(image_path)
        encoded_image = ocr_client.encode_image_to_base64(resized_image_data)

        # Invoke OCR function
        captcha_text = ocr_client.invoke_gpt4o_ocr(encoded_image)

        # Output the recognized text
        if captcha_text:
            print("识别出的验证码是：", captcha_text)
        else:
            print("验证码识别失败，请稍后重试。")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
