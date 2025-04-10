# Text-to-Hand-Written Bot

A Telegram bot that can convert text to handwritten PDFs and edit screenshots with text replacement features.

## Features

- Convert text to handwritten PDF
- Edit screenshots:
  - Replace text in screenshots
  - Add text to screenshots
- Image processing:
  - Compress images
  - Convert between PNG and JPG
- PDF operations:
  - Compress PDFs
  - Merge PDFs
  - Convert between Word and PDF
  - Add watermarks

## Setup

1. Install Python 3.7 or higher
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Install Tesseract OCR:
   - Windows: Download from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
   - Linux: `sudo apt-get install tesseract-ocr`
4. Set your bot token in `bot.py`
5. Run the bot:
   ```bash
   python bot.py
   ```

## Usage

1. Start the bot with /start
2. Choose from available options:
   - ✍️ Handwritten PDF
   - 🖼️ Screenshot Edit
   - Other file operations
3. Follow the bot's instructions

## Requirements

See requirements.txt for full list of dependencies. 