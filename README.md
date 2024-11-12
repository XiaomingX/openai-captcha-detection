# openai-captcha-detection

## 项目简介

`openai-captcha-detection` 是一个使用 OpenAI 进行验证码识别的工具。通过调用 OpenAI 的 API，这个项目可以实现对复杂验证码图片的文本识别，帮助开发者在验证码处理场景中进行自动化操作。

## 功能特点
- 利用 OpenAI 的 GPT-4 模型进行 OCR（光学字符识别），可以识别各种类型的验证码。
- 提供简单易用的 API 接口，方便在其他项目中集成使用。

## 环境准备

在使用此项目前，请确保已经安装好以下工具：
- Python 3.7 及以上版本
- pip 包管理工具

## 安装与使用

### 克隆项目并运行验证
1. 克隆仓库并进入项目目录：
    ```sh
    git clone --depth 1 https://github.com/zgimszhd61/openai-capcha-detection
    cd openai-capcha-detection
    ```

2. 设置 OpenAI API 密钥：
    ```sh
    export OPENAI_API_KEY=[你的API_KEY]
    ```

3. 安装所需依赖包：
    ```sh
    pip install -r requirements.txt
    ```

4. 运行验证码识别脚本：
    ```sh
    python3 src/gpt4_ocr_demo.py
    ```

### 在其他项目中集成使用
你可以在自己的项目中集成 `openai-captcha-detection` 来实现验证码识别。以下是一个使用示例：

```python
from gpt4_ocr_demo import OCRClient

def recognize_captcha(image_path):
    # 创建 OCRClient 实例
    ocr_client = OCRClient()

    # 将图片编码为 base64 格式
    encoded_image = ocr_client.encode_image_to_base64(image_path)

    # 调用 GPT-4 OCR 函数进行识别
    captcha_text = ocr_client.invoke_gpt4_ocr(encoded_image)

    return captcha_text

if __name__ == "__main__":
    image_path = "226md.png"
    recognized_text = recognize_captcha(image_path)
    print("识别出的验证码是：", recognized_text)
```

## 项目结构
- `src/gpt4_ocr_demo.py`：封装了与 OpenAI API 交互的客户端类，包括图像编码与验证码识别的主要功能。演示如何使用 GPT-4 模型进行验证码识别的脚本。
- 
## 注意事项
- 请确保您在使用 OpenAI API 时具有有效的 API Key，并注意使用频率以免超出额度。
- 识别效果取决于验证码的复杂程度以及 GPT-4 模型的能力，某些复杂验证码可能会存在识别错误的情况。

## 未来规划
- 提升对复杂验证码类型的识别精度。
- 增加更多的 API 支持，以更好地适应多样化的验证码类型。
- 提供更便捷的命令行界面以供用户直接使用。

## 许可证
此项目遵循 MIT 许可证，详细内容请查看 [LICENSE](LICENSE) 文件。