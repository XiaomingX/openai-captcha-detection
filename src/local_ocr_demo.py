import asyncio
from typing import Tuple, Optional
import pytesseract
from PIL import Image, ImageFilter, ImageOps

class ThreeAntiCaptchaImageSolver:
    def __init__(self, tesseract_cmd: str = "tesseract", psm: int = 8):
        # 设置 Tesseract OCR 可执行文件的路径
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self.psm = psm  # OCR 识别模式

    def preprocess_image(self, image_path: str) -> Image.Image:
        """
        对图像进行预处理：灰度化、二值化、锐化等
        """
        image = Image.open(image_path)
        image = ImageOps.grayscale(image)  # 转为灰度图像
        image = image.filter(ImageFilter.MedianFilter())  # 去噪
        image = ImageOps.autocontrast(image)  # 自动对比度调整
        threshold = 128  # 二值化阈值
        image = image.point(lambda p: p > threshold and 255)  # 二值化
        return image

    async def solve(self, image_path: str, custom_config: Optional[str] = None) -> Tuple[str, bool]:
        """
        使用 OCR 识别验证码文本
        """
        try:
            # 预处理图像
            image = self.preprocess_image(image_path)
            
            # 使用 pytesseract 识别验证码文本
            config = custom_config if custom_config else f"--psm {self.psm}"
            captcha_text = pytesseract.image_to_string(image, config=config)

            # 去除空白符并检查长度是否符合预期
            captcha_text = captcha_text.strip()
            if captcha_text:
                return captcha_text, True
            else:
                return "Recognition failed - No text detected", False

        except Exception as err:
            return f"An unexpected error occurred: {err}", False

    async def report_bad(self, reason: str = "Low accuracy") -> Tuple[str, bool]:
        """
        模拟报告错误验证码，记录日志
        """
        # 可以将错误原因记录在日志中，方便后续分析
        with open("error_log.txt", "a") as log_file:
            log_file.write(f"Bad captcha reported: {reason}\n")
        return "Logged error report", True


async def main():
    image_path = "img/226md.png"
    solver = ThreeAntiCaptchaImageSolver(psm=8)
    result, success = await solver.solve(image_path)
    print(f"Result: {result}, Success: {success}")

    # 示例：在验证码识别错误的情况下报告
    if not success:
        await solver.report_bad("Recognition did not meet expected accuracy")


if __name__ == "__main__":
    asyncio.run(main())
