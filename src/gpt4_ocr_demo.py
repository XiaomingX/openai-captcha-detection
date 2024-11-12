import base64
from openai import OpenAI
from dotenv import load_dotenv
import os

## 便宜OCR识别验证码

class OCRClient:
    def __init__(self):
        self._load_environment()
        self.client = self._initialize_openai_client()

    def _load_environment(self):
        """Load environment variables from a .env file."""
        load_dotenv()

    def _initialize_openai_client(self):
        """Initialize and return an OpenAI client."""
        return OpenAI()

    def encode_image_to_base64(self, image_path):
        """Read image from the given path and return base64 encoded string."""
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
        return base64.b64encode(image_data).decode('utf-8')

    def invoke_gpt4o_ocr(self, encoded_image):
        """Invoke GPT-4o-mini to perform OCR on the provided base64 encoded image."""
        completion = self.client.chat.completions.create(
            model="gpt-4o-mini",
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
        result = completion.choices[0].message.content.replace("-","").strip()
        return result

def main():
    # Create an OCR client instance
    ocr_client = OCRClient()

    # Set image path
    image_path = 'img/image.png'
    # image_path = 'img/226md.png'
    
    # Encode the image to base64
    encoded_image = ocr_client.encode_image_to_base64(image_path)

    # Invoke OCR function
    captcha_text = ocr_client.invoke_gpt4o_ocr(encoded_image)

    # Output the recognized text
    print("识别出的验证码是：", captcha_text)

if __name__ == "__main__":
    main()