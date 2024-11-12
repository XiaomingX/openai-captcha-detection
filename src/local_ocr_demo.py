import asyncio
from typing import Tuple
import pytesseract
from PIL import Image

## comemnt:免费，但是效果不好. 识别率很低

class ThreeAntiCaptchaImageSolver:
    def __init__(self, tesseract_cmd: str = "tesseract"):
        # 设置 Tesseract OCR 可执行文件的路径
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    async def solve(self, image_path: str) -> Tuple[str, bool]:
        try:
            # 使用 PIL 打开图像
            image = Image.open(image_path)

            # 使用 pytesseract 识别验证码文本
            captcha_text = pytesseract.image_to_string(image, config="--psm 8")

            # 去除空白符并检查长度是否符合预期
            captcha_text = captcha_text.strip()
            return captcha_text, True

        except Exception as err:
            return f"An unexpected error occurred: {err}", False

    async def report_bad(self, image_path: str) -> Tuple[str, bool]:
        # 本地验证码验证的场景中，报告错误的验证码没有实际意义，但可以记录日志或采取其他操作
        return "Reporting not applicable for local captcha solving", False


async def main():
    image_path = "img/image.png"
    solver = ThreeAntiCaptchaImageSolver()
    result, success = await solver.solve(image_path)
    print(f"Result: {result}, Success: {success}")


if __name__ == "__main__":
    asyncio.run(main())