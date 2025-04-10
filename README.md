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

## Deployment to Railway

1. Create a Railway account at [railway.app](https://railway.app)
2. Install Railway CLI:
   ```bash
   npm i -g @railway/cli
   ```
3. Link your project:
   ```bash
   railway link
   ```
4. Deploy your project:
   ```bash
   railway up
   ```

### Environment Variables

Set these environment variables in Railway dashboard:
- `BOT_TOKEN`: Your Telegram bot token
- `TESSERACT_PATH`: Path to Tesseract OCR (default: `/usr/bin/tesseract` for Linux)

## Usage

1. Start the bot with /start
2. Choose from available options:
   - ✍️ Handwritten PDF
   - 🖼️ Screenshot Edit
   - Other file operations
3. Follow the bot's instructions

## Requirements

See requirements.txt for full list of dependencies. 