import os
import uuid
import zipfile
import telebot
from telebot import types
from fpdf import FPDF
from PIL import Image, ImageDraw, ImageFont
from docx2pdf import convert
from pdf2docx import Converter
from PyPDF2 import PdfReader, PdfWriter, PdfMerger
import pytesseract
import sys
import cv2
import numpy as np

# Set Tesseract path and check if it exists
TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
if not os.path.exists(TESSERACT_PATH):
    print(f"Error: Tesseract not found at {TESSERACT_PATH}")
    print("Please install Tesseract-OCR from: https://github.com/UB-Mannheim/tesseract/wiki")
    sys.exit(1)

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

bot = telebot.TeleBot("7460209179:AAFFw9HB95heL2NVj8fX0z1dxML56fVqTyc")

FONT_PATH = os.path.join(os.path.dirname(__file__), "QECarolineMutiboko.ttf")
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LINES_PER_PAGE = 25
FONT_SIZE = 20
PAGE_WIDTH, PAGE_HEIGHT = 595, 842
MARGIN = 50
LINE_HEIGHT = 30
MAX_LINE_WIDTH = PAGE_WIDTH - (2 * MARGIN)

user_context = {}
user_temp_files = {}
user_settings = {}
user_states = {}  # To track user states for screenshot editing

class HandwrittenPDF(FPDF):
    def header(self):
        self.set_font("Arial", size=12)
        self.cell(0, 10, '', 0, 1, 'C')

def get_text_width(text, font):
    bbox = font.getbbox(text)
    return bbox[2] - bbox[0]

def split_text_to_fit_width(text, font, max_width):
    words = text.split()
    lines, current_line, current_width = [], [], 0
    for word in words:
        word_width = get_text_width(word + ' ', font)
        if current_width + word_width <= max_width:
            current_line.append(word)
            current_width += word_width
        else:
            lines.append(' '.join(current_line))
            current_line = [word]
            current_width = word_width
    if current_line:
        lines.append(' '.join(current_line))
    return lines

def create_handwritten_pdf(text, output_path):
    pdf = HandwrittenPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

    original_lines = text.splitlines()
    processed_lines = []
    for line in original_lines:
        if line.strip():
            processed_lines.extend(split_text_to_fit_width(line, font, MAX_LINE_WIDTH))
        else:
            processed_lines.append('')

    pages = [processed_lines[i:i + LINES_PER_PAGE] for i in range(0, len(processed_lines), LINES_PER_PAGE)]

    for page_num, page_lines in enumerate(pages):
        img = Image.new('RGB', (PAGE_WIDTH, PAGE_HEIGHT), color='white')
        draw = ImageDraw.Draw(img)
        y = MARGIN
        for line in page_lines:
            draw.text((MARGIN, y), line, font=font, fill='black')
            y += LINE_HEIGHT
        image_path = os.path.join(OUTPUT_DIR, f"temp_page_{page_num}.jpg")
        img.save(image_path)
        pdf.add_page()
        pdf.image(image_path, x=0, y=0, w=210, h=297)
        os.remove(image_path)

    pdf.output(output_path)

def compress_pdf(input_path, output_path):
    reader = PdfReader(input_path)
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    with open(output_path, "wb") as f:
        writer.write(f)

def compress_image(input_path, output_path, quality=40):
    img = Image.open(input_path)
    original_size = os.path.getsize(input_path)
    img.save(output_path, optimize=True, quality=quality)
    compressed_size = os.path.getsize(output_path)
    return original_size, compressed_size

def add_watermark_pdf(input_path, output_path, watermark_text="Watermark"):
    reader = PdfReader(input_path)
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    with open(output_path, "wb") as f:
        writer.write(f)

def add_watermark_image(input_path, output_path, watermark_text="Watermark"):
    img = Image.open(input_path).convert("RGBA")
    txt_layer = Image.new("RGBA", img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(txt_layer)
    font = ImageFont.truetype(FONT_PATH, 20)
    draw.text((10, 10), watermark_text, fill=(0, 0, 0, 100), font=font)
    watermarked = Image.alpha_composite(img, txt_layer)
    watermarked.convert("RGB").save(output_path)

def merge_pdfs(file_paths, output_path):
    merger = PdfMerger()
    for path in file_paths:
        merger.append(path)
    with open(output_path, "wb") as f:
        merger.write(f)

def enhance_image_for_ocr(image_path):
    # Read image using opencv
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to preprocess the image
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    # Apply dilation to connect text components
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    gray = cv2.dilate(gray, kernel, iterations=1)
    
    # Write the grayscale image temporarily
    temp_path = image_path.replace('.png', '_enhanced.png')
    cv2.imwrite(temp_path, gray)
    return temp_path

def analyze_text_blocks(image_path):
    try:
        # Enhance image first
        enhanced_path = enhance_image_for_ocr(image_path)
        
        # Extract text using Tesseract
        img = Image.open(enhanced_path).convert("RGB")
        data = safe_ocr(enhanced_path)
        
        if not data:
            raise ValueError("Failed to extract text from image")
        
        # Group words into meaningful blocks
        blocks = []
        current_block = []
        current_y = None
        line_height_threshold = 10  # Adjust based on your needs
        
        for i, word in enumerate(data['text']):
            if word.strip():
                if current_y is None:
                    current_y = data['top'][i]
                    current_block = [word.strip()]
                else:
                    # If word is on roughly the same line
                    if abs(data['top'][i] - current_y) <= line_height_threshold:
                        current_block.append(word.strip())
                    else:
                        # New line detected
                        if current_block:
                            blocks.append({
                                "text": " ".join(current_block),
                                "position": (
                                    min(data['left'][i-len(current_block):i]),
                                    current_y,
                                    max(data['left'][i-len(current_block):i]) + max(data['width'][i-len(current_block):i]),
                                    current_y + max(data['height'][i-len(current_block):i])
                                )
                            })
                        current_block = [word.strip()]
                        current_y = data['top'][i]
        
        # Add the last block
        if current_block:
            blocks.append({
                "text": " ".join(current_block),
                "position": (
                    data['left'][len(data['text'])-len(current_block)],
                    current_y,
                    data['left'][len(data['text'])-1] + data['width'][len(data['text'])-1],
                    current_y + data['height'][len(data['text'])-1]
                )
            })
        
        # Clean up enhanced image
        if os.path.exists(enhanced_path):
            os.remove(enhanced_path)
            
        return blocks
        
    except Exception as e:
        if os.path.exists(enhanced_path):
            os.remove(enhanced_path)
        raise ValueError(f"Error analyzing image: {str(e)}")

def extract_text_blocks(image_path):
    try:
        blocks = analyze_text_blocks(image_path)
        if not blocks:
            raise ValueError("No text found in image")
        return blocks
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")

def replace_text_in_screenshot(image_path, old_text, new_text, output_path, font_size=24):
    try:
        img = Image.open(image_path)
        blocks = analyze_text_blocks(image_path)
        
        if not blocks:
            raise ValueError("No text found in image")
        
        # Create a copy of the image for drawing
        img_array = np.array(img)
        
        # Make text matching more flexible
        old_text = old_text.strip()
        found = False
        
        # First try exact match
        for block in blocks:
            if block['text'].strip() == old_text:
                x1, y1, x2, y2 = block['position']
                
                # Get the region height and calculate font size
                text_height = y2 - y1
                font_size_adjusted = int(text_height * 0.8)  # 80% of block height
                
                try:
                    font = ImageFont.truetype("arial.ttf", font_size_adjusted)
                except:
                    try:
                        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size_adjusted)
                    except:
                        font = ImageFont.load_default()
                
                # Create a temporary image for the region
                region_width = x2 - x1
                region_height = y2 - y1
                temp_img = Image.new('RGBA', (region_width, region_height), (0, 0, 0, 0))
                temp_draw = ImageDraw.Draw(temp_img)
                
                # Get text size
                text_bbox = font.getbbox(new_text)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                # Calculate position to center text
                x_pos = (region_width - text_width) // 2
                y_pos = (region_height - text_height) // 2
                
                # Draw text on temporary image
                temp_draw.text((x_pos, y_pos), new_text, font=font, fill="black")
                
                # Convert image to numpy array
                img_array = np.array(img)
                temp_array = np.array(temp_img)
                
                # Create a mask for the old text
                mask = np.zeros((region_height, region_width), dtype=np.uint8)
                cv2.putText(mask, old_text, (0, region_height-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 2)
                
                # Get background color
                bg_color = get_dominant_color(img, x1, y1, x2, y2)
                
                # Replace old text region with background color
                img_array[y1:y2, x1:x2][mask > 0] = bg_color
                
                # Overlay new text
                alpha = temp_array[:, :, 3] / 255.0
                for c in range(3):
                    img_array[y1:y2, x1:x2, c] = (
                        (1 - alpha) * img_array[y1:y2, x1:x2, c] +
                        alpha * temp_array[:, :, c]
                    )
                
                found = True
                break
        
        if not found:
            # Try case-insensitive match
            old_text_lower = old_text.lower()
            for block in blocks:
                if block['text'].strip().lower() == old_text_lower:
                    x1, y1, x2, y2 = block['position']
                    text_height = y2 - y1
                    font_size_adjusted = int(text_height * 0.8)
                    
                    try:
                        font = ImageFont.truetype("arial.ttf", font_size_adjusted)
                    except:
                        try:
                            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size_adjusted)
                        except:
                            font = ImageFont.load_default()
                    
                    # Create temporary image for the region
                    region_width = x2 - x1
                    region_height = y2 - y1
                    temp_img = Image.new('RGBA', (region_width, region_height), (0, 0, 0, 0))
                    temp_draw = ImageDraw.Draw(temp_img)
                    
                    # Get text size
                    text_bbox = font.getbbox(new_text)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                    
                    # Calculate position to center text
                    x_pos = (region_width - text_width) // 2
                    y_pos = (region_height - text_height) // 2
                    
                    # Draw text on temporary image
                    temp_draw.text((x_pos, y_pos), new_text, font=font, fill="black")
                    
                    # Convert image to numpy array
                    temp_array = np.array(temp_img)
                    
                    # Create a mask for the old text
                    mask = np.zeros((region_height, region_width), dtype=np.uint8)
                    cv2.putText(mask, block['text'].strip(), (0, region_height-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 2)
                    
                    # Get background color
                    bg_color = get_dominant_color(img, x1, y1, x2, y2)
                    
                    # Replace old text region with background color
                    img_array[y1:y2, x1:x2][mask > 0] = bg_color
                    
                    # Overlay new text
                    alpha = temp_array[:, :, 3] / 255.0
                    for c in range(3):
                        img_array[y1:y2, x1:x2, c] = (
                            (1 - alpha) * img_array[y1:y2, x1:x2, c] +
                            alpha * temp_array[:, :, c]
                        )
                    
                    found = True
                    break
        
        if not found:
            # Provide available text blocks for reference
            available_blocks = [block['text'].strip() for block in blocks]
            text_list = "\n".join([f"'{text}'" for text in available_blocks])
            raise ValueError(f"Could not find text '{old_text}' in the image.\n\nAvailable text blocks:\n{text_list}")
        
        # Save the modified image
        Image.fromarray(img_array).save(output_path)
        
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")

def get_dominant_color(img, x1, y1, x2, y2):
    # Get a region of the image
    region = img.crop((x1, y1, x2, y2))
    # Convert to RGB if not already
    region = region.convert('RGB')
    # Get the average color of the region
    avg_color = [int(sum(x)/len(x)) for x in zip(*region.getdata())]
    return tuple(avg_color)

# Add error handling for OCR operations
def safe_ocr(image_path):
    try:
        return pytesseract.image_to_data(Image.open(image_path).convert("RGB"), output_type=pytesseract.Output.DICT)
    except Exception as e:
        print(f"OCR Error: {str(e)}")
        return None

@bot.message_handler(commands=['start'])
def send_welcome(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    markup.add(
        types.KeyboardButton("✍️ Handwritten PDF"),
        types.KeyboardButton("🖼️ Screenshot Edit"),
        types.KeyboardButton("📋 Main Menu")
    )
    bot.send_message(message.chat.id, "Welcome! Choose an option:", reply_markup=markup)
    show_main_menu(message.chat.id, "Or use the inline menu below:")

@bot.message_handler(func=lambda message: message.text == "🖼️ Screenshot Edit")
def handle_screenshot_edit_menu(message):
    chat_id = message.chat.id
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    markup.add(
        types.KeyboardButton("✏️ Add Text"),
        types.KeyboardButton("🔄 Replace Text"),
        types.KeyboardButton("📋 Main Menu")
    )
    # Initialize user context for screenshot editing
    user_context[chat_id] = 'screenshot_edit'
    bot.send_message(message.chat.id, "Choose a screenshot editing option:", reply_markup=markup)

@bot.message_handler(func=lambda message: message.text == "✏️ Add Text")
def handle_add_text_request(message):
    chat_id = message.chat.id
    user_states[chat_id] = "waiting_for_screenshot_add"
    user_context[chat_id] = 'screenshot_add'
    if chat_id not in user_temp_files:
        user_temp_files[chat_id] = []
    bot.send_message(message.chat.id, "📸 Please send the screenshot where you want to add text.")

@bot.message_handler(func=lambda message: message.text == "🔄 Replace Text")
def handle_replace_text_request(message):
    chat_id = message.chat.id
    user_states[chat_id] = "waiting_for_screenshot_replace"
    user_context[chat_id] = 'screenshot_replace'
    if chat_id not in user_temp_files:
        user_temp_files[chat_id] = []
    bot.send_message(message.chat.id, "📸 Please send the screenshot where you want to replace text.")

@bot.message_handler(func=lambda message: message.text == "📋 Main Menu")
def handle_main_menu(message):
    send_welcome(message)

@bot.message_handler(content_types=['photo'])
def handle_screenshot(message):
    chat_id = message.chat.id
    state = user_states.get(chat_id)
    
    if state not in ["waiting_for_screenshot_add", "waiting_for_screenshot_replace"]:
        bot.reply_to(message, "❌ Please select an editing option first.")
        return

    try:
        # Download the photo
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        
        # Save the screenshot
        screenshot_path = os.path.join(OUTPUT_DIR, f"screenshot_{chat_id}.png")
        with open(screenshot_path, 'wb') as f:
            f.write(downloaded_file)
        
        # Store the screenshot path
        user_temp_files[chat_id] = screenshot_path
        
        if state == "waiting_for_screenshot_add":
            user_states[chat_id] = "waiting_for_add_text"
            bot.reply_to(message, "✍️ Now send the text you want to add to the screenshot.")
        
        elif state == "waiting_for_screenshot_replace":
            try:
                blocks = analyze_text_blocks(screenshot_path)
                if not blocks:
                    bot.reply_to(message, "❌ No text found in the screenshot. Make sure the text is clear and readable.")
                    if os.path.exists(screenshot_path):
                        os.remove(screenshot_path)
                    return
                
                # Format text blocks nicely with analysis
                text_blocks = []
                for block in blocks:
                    text = block['text'].strip()
                    if text:
                        text_blocks.append(text)
                
                if not text_blocks:
                    bot.reply_to(message, "❌ No clear text found in the screenshot. Make sure the text is clear and readable.")
                    if os.path.exists(screenshot_path):
                        os.remove(screenshot_path)
                    return
                
                # Format text blocks nicely
                text_list = "\n".join([f"{i+1}. '{text}'" for i, text in enumerate(text_blocks)])
                bot.reply_to(message, 
                    f"📝 Found these text blocks:\n\n{text_list}\n\n"
                    f"✍️ Please copy and paste the exact text you want to replace from above.\n\n"
                    f"💡 Tips:\n"
                    f"• Make sure to copy the exact text including spaces\n"
                    f"• Text matching is case-sensitive\n"
                    f"• If text isn't detected correctly, try taking a clearer screenshot")
                user_states[chat_id] = "waiting_for_old_text"
                
            except Exception as e:
                bot.reply_to(message, f"❌ Error processing screenshot: {str(e)}")
                if os.path.exists(screenshot_path):
                    os.remove(screenshot_path)
                
    except Exception as e:
        bot.reply_to(message, f"❌ Error handling screenshot: {str(e)}")
        if chat_id in user_states:
            del user_states[chat_id]

@bot.message_handler(func=lambda message: user_states.get(message.chat.id) == "waiting_for_old_text")
def handle_old_text(message):
    chat_id = message.chat.id
    screenshot_path = user_temp_files.get(chat_id)
    
    if not screenshot_path or not os.path.exists(screenshot_path):
        bot.reply_to(message, "❌ Please send the screenshot first.")
        return
        
    try:
        # Verify the text exists in the image
        blocks = extract_text_blocks(screenshot_path)
        found_texts = [block['text'].strip() for block in blocks if block['text'].strip()]
        
        # Try to find exact match first
        text_exists = message.text.strip() in found_texts
        
        # If no exact match, try case-insensitive match
        if not text_exists:
            text_exists = any(text.lower() == message.text.strip().lower() for text in found_texts)
        
        if not text_exists:
            # Show available text blocks with better formatting
            text_list = "\n".join([f"{i+1}. '{text}'" for i, text in enumerate(found_texts)])
            bot.reply_to(message, 
                f"❌ Could not find '{message.text}' in the image.\n\n"
                f"Available text blocks:\n{text_list}\n\n"
                f"Please copy and paste one of the exact text blocks shown above.")
            return
            
        user_context[chat_id] = {"old_text": message.text.strip()}
        user_states[chat_id] = "waiting_for_new_text"
        bot.reply_to(message, "✅ Text found! Now send the new text you want to replace it with.")
        
    except Exception as e:
        bot.reply_to(message, f"❌ Error: {str(e)}")
        if chat_id in user_states:
            del user_states[chat_id]

@bot.message_handler(func=lambda message: user_states.get(message.chat.id) == "waiting_for_new_text")
def handle_new_text(message):
    chat_id = message.chat.id
    screenshot_path = user_temp_files.get(chat_id)
    old_text = user_context.get(chat_id, {}).get("old_text")
    
    if not screenshot_path or not old_text or not os.path.exists(screenshot_path):
        bot.reply_to(message, "❌ Please start over.")
        return
    
    try:
        output_path = os.path.join(OUTPUT_DIR, f"edited_screenshot_{chat_id}.png")
        replace_text_in_screenshot(screenshot_path, old_text, message.text, output_path)
        
        with open(output_path, 'rb') as f:
            bot.send_photo(chat_id, f, caption="✅ Here's your edited screenshot!")
        
        # Cleanup
        if os.path.exists(screenshot_path):
            os.remove(screenshot_path)
        if os.path.exists(output_path):
            os.remove(output_path)
        if chat_id in user_states:
            del user_states[chat_id]
        if chat_id in user_temp_files:
            del user_temp_files[chat_id]
        if chat_id in user_context:
            del user_context[chat_id]
        
    except Exception as e:
        bot.reply_to(message, f"❌ Error replacing text: {str(e)}")
        # Show available text blocks on error
        try:
            blocks = extract_text_blocks(screenshot_path)
            found_texts = [block['text'].strip() for block in blocks if block['text'].strip()]
            text_list = "\n".join([f"{i+1}. '{text}'" for i, text in enumerate(found_texts)])
            bot.reply_to(message, 
                f"Available text blocks:\n{text_list}\n\n"
                f"Please make sure to copy and paste one of these exact text blocks.")
        except:
            pass

def show_main_menu(chat_id, text):
    markup = types.InlineKeyboardMarkup(row_width=2)
    markup.add(
        types.InlineKeyboardButton("📝 Text to Handwritten PDF", callback_data='handwritten'),
        types.InlineKeyboardButton("📄 Compress PDF", callback_data='compress_pdf'),
        types.InlineKeyboardButton("📉 Compress Image", callback_data='compress_image'),
        types.InlineKeyboardButton("🔁 Word to PDF", callback_data='word_to_pdf'),
        types.InlineKeyboardButton("🔁 PDF to Word", callback_data='pdf_to_word'),
        types.InlineKeyboardButton("🖼 JPG to PNG", callback_data='jpg_to_png'),
        types.InlineKeyboardButton("🖼 PNG to JPG", callback_data='png_to_jpg'),
        types.InlineKeyboardButton("📚 Merge PDFs", callback_data='merge_pdfs'),
        types.InlineKeyboardButton("🔧 Toggle Watermark", callback_data='toggle_watermark'),
        types.InlineKeyboardButton("🔧 Toggle Compression", callback_data='toggle_compression')
    )
    bot.send_message(chat_id, text, reply_markup=markup)

@bot.callback_query_handler(func=lambda call: True)
def handle_menu_selection(call):
    chat_id = call.message.chat.id
    if call.data == 'main_menu':
        return show_main_menu(chat_id, "📋 Main menu:")

    if call.data == 'toggle_watermark':
        user_settings[chat_id] = user_settings.get(chat_id, {})
        current = user_settings[chat_id].get("watermark", False)
        user_settings[chat_id]["watermark"] = not current
        return bot.send_message(chat_id, f"✅ Watermark {'enabled' if not current else 'disabled'}")

    if call.data == 'toggle_compression':
        user_settings[chat_id] = user_settings.get(chat_id, {})
        current = user_settings[chat_id].get("compress", False)
        user_settings[chat_id]["compress"] = not current
        return bot.send_message(chat_id, f"✅ Compression {'enabled' if not current else 'disabled'}")

    user_context[chat_id] = call.data
    user_temp_files[chat_id] = []
    user_settings[chat_id] = user_settings.get(chat_id, {"watermark": False, "compress": False})

    msg = "📤 Send the required file(s). You can send multiple files."
    if call.data == 'handwritten':
        msg = "📤 Send a `.txt` file."
    bot.send_message(chat_id, msg)

    markup = types.InlineKeyboardMarkup()
    markup.add(types.InlineKeyboardButton("📋 Menu", callback_data='main_menu'))
    bot.send_message(chat_id, "📋 Use the menu to switch tasks:", reply_markup=markup)

@bot.message_handler(content_types=['document'])
def handle_files(message):
    chat_id = message.chat.id
    
    # Initialize user context if not exists
    if chat_id not in user_context:
        bot.reply_to(message, "❌ Please select an option from the menu first.")
        return
        
    context = user_context.get(chat_id)
    settings = user_settings.get(chat_id, {})
    
    # Initialize temp files list if not exists
    if chat_id not in user_temp_files:
        user_temp_files[chat_id] = []
    
    try:
        file_info = bot.get_file(message.document.file_id)
        file_data = bot.download_file(file_info.file_path)
        ext = os.path.splitext(message.document.file_name)[-1].lower()
        file_path = os.path.join(OUTPUT_DIR, f"{uuid.uuid4()}{ext}")
        
        with open(file_path, 'wb') as f:
            f.write(file_data)

        user_temp_files[chat_id].append(file_path)

        if context == 'handwritten':
            if not file_path.endswith(".txt"):
                bot.reply_to(message, "❌ Please send a `.txt` file.")
                return
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                out_path = os.path.join(OUTPUT_DIR, 'handwritten.pdf')
                create_handwritten_pdf(text, out_path)
                with open(out_path, 'rb') as f:
                    bot.send_document(chat_id, f)
                # Cleanup
                os.remove(out_path)
            except Exception as e:
                bot.reply_to(message, f"❌ Error creating handwritten PDF: {str(e)}")

        elif context == 'merge_pdfs':
            if len(user_temp_files[chat_id]) >= 2:
                try:
                    out_path = os.path.join(OUTPUT_DIR, f"merged_{uuid.uuid4()}.pdf")
                    merge_pdfs(user_temp_files[chat_id], out_path)
                    with open(out_path, 'rb') as f:
                        bot.send_document(chat_id, f)
                    # Cleanup
                    os.remove(out_path)
                    user_temp_files[chat_id] = []
                except Exception as e:
                    bot.reply_to(message, f"❌ Error merging PDFs: {str(e)}")
            else:
                bot.send_message(chat_id, "📎 Send one more PDF to merge.")

        elif context in ['word_to_pdf', 'pdf_to_word', 'compress_pdf', 'compress_image', 'jpg_to_png', 'png_to_jpg']:
            try:
                out_path = file_path
                if context == 'word_to_pdf':
                    out_path = file_path.replace(".docx", ".pdf")
                    convert(file_path, out_path)
                elif context == 'pdf_to_word':
                    out_path = file_path.replace(".pdf", ".docx")
                    cv = Converter(file_path)
                    cv.convert(out_path, start=0, end=None)
                    cv.close()
                elif context == 'compress_pdf':
                    out_path = file_path.replace(".pdf", "_compressed.pdf")
                    compress_pdf(file_path, out_path)
                elif context == 'compress_image':
                    out_path = file_path.replace(ext, f"_compressed{ext}")
                    compress_image(file_path, out_path)
                elif context == 'jpg_to_png':
                    img = Image.open(file_path)
                    out_path = file_path.replace(".jpg", ".png")
                    img.save(out_path, 'PNG')
                elif context == 'png_to_jpg':
                    img = Image.open(file_path)
                    out_path = file_path.replace(".png", ".jpg")
                    img.convert("RGB").save(out_path, 'JPEG')

                if settings.get("watermark"):
                    if out_path.endswith(".pdf"):
                        wm_path = out_path.replace(".pdf", "_wm.pdf")
                        add_watermark_pdf(out_path, wm_path)
                        out_path = wm_path
                    elif out_path.endswith(".jpg") or out_path.endswith(".png"):
                        wm_path = out_path.replace(ext, f"_wm{ext}")
                        add_watermark_image(out_path, wm_path)
                        out_path = wm_path

                with open(out_path, 'rb') as f:
                    bot.send_document(chat_id, f)
                    
            except Exception as e:
                bot.reply_to(message, f"❌ Error processing file: {str(e)}")
            finally:
                # Cleanup temporary files
                try:
                    os.remove(file_path)
                    if out_path != file_path:
                        os.remove(out_path)
                except:
                    pass
                
    except Exception as e:
        bot.reply_to(message, f"❌ Error handling file: {str(e)}")
    finally:
        # Clear user context after processing
        if chat_id in user_temp_files:
            for temp_file in user_temp_files[chat_id]:
                try:
                    os.remove(temp_file)
                except:
                    pass
            user_temp_files[chat_id] = []

# Start bot with error handling
while True:
    try:
        bot.infinity_polling(timeout=60, long_polling_timeout=60)
    except Exception as e:
        print(f"Bot polling error: {e}")
        time.sleep(2)
        continue
