import os
import re
import argparse
import zipfile
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import html2text
from urllib.parse import unquote, urlparse
from PIL import Image, ImageOps
import io

def resize_and_convert_to_png(image_data, output_path, max_width=1000):
    with Image.open(io.BytesIO(image_data)) as img:
        img = img.convert('RGBA')
        
        # Resize if width is greater than max_width
        if img.width > max_width:
            ratio = max_width / float(img.width)
            height = int(float(img.height) * ratio)
            img = img.resize((max_width, height), Image.LANCZOS)
        
        # Calculate target dimensions for 4:3 ratio
        target_ratio = 4 / 3
        current_ratio = img.width / img.height
        
        if current_ratio > target_ratio:
            # Image is wider than 4:3, add vertical padding
            new_height = int(img.width / target_ratio)
            new_width = img.width
        else:
            # Image is taller than 4:3, add horizontal padding
            new_width = int(img.height * target_ratio)
            new_height = img.height
        
        # Create a white background image with the target size
        background = Image.new('RGBA', (new_width, new_height), (255, 255, 255, 255))
        
        # Calculate position to paste the original image
        paste_x = (new_width - img.width) // 2
        paste_y = (new_height - img.height) // 2
        
        # Paste the original image onto the white background
        background.paste(img, (paste_x, paste_y), img)
        
        # Convert to RGB before saving as PNG
        background = background.convert('RGB')
        background.save(output_path, 'PNG')

def extract_images_from_epub(epub_path, output_folder):
    with zipfile.ZipFile(epub_path, 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            if file_info.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                original_filename = os.path.basename(file_info.filename)
                png_filename = os.path.splitext(original_filename)[0] + '.png'
                png_path = os.path.join(output_folder, png_filename)
                
                image_data = zip_ref.read(file_info.filename)
                resize_and_convert_to_png(image_data, png_path)
                yield png_filename

def get_cover_image(book, output_folder):
    cover_id = book.get_metadata('OPF', 'cover')
    if cover_id:
        cover_id = cover_id[0][1]['content']
        cover_item = book.get_item_with_id(cover_id)
        if cover_item:
            cover_filename = 'cover.png'
            cover_path = os.path.join(output_folder, cover_filename)
            resize_and_convert_to_png(cover_item.content, cover_path)
            return cover_filename
    return None

def is_valid_image_src(src):
    return (src.startswith('http') or 
            src.startswith('www.') or 
            any(src.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif']))

def is_valid_link(href):
    parsed = urlparse(href)
    return bool(parsed.netloc) and parsed.scheme in ('http', 'https')

def clean_html_content(soup, extracted_images):
    # Handle links and images
    for a in soup.find_all('a'):
        href = a.get('href', '')
        if is_valid_link(href):
            # Keep valid web links
            a.replace_with(f"[{a.text}]({href})")
        else:
            # Remove other links
            a.replace_with(a.text)
    
    # Handle images
    for img in soup.find_all('img'):
        src = img.get('src', '')
        alt = img.get('alt', '')
        if is_valid_image_src(src):
            original_filename = os.path.basename(src)
            png_filename = os.path.splitext(original_filename)[0] + '.png'
            if png_filename in extracted_images:
                # Local image file, use only the filename
                img.replace_with(f'![{alt}]({png_filename})')
            else:
                # Web image
                img.replace_with(f'![{alt}]({src})')
        else:
            # Remove invalid image tags
            img.decompose()
    
    return str(soup)

def epub_to_markdown(epub_folder, output_folder, preserve_structure=False):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, _, files in os.walk(epub_folder):
        for filename in files:
            if filename.endswith('.epub'):
                epub_path = os.path.join(root, filename)
                book = epub.read_epub(epub_path)
                
                # Create a folder for this EPUB's output
                epub_name = os.path.splitext(filename)[0]
                if preserve_structure:
                    relative_path = os.path.relpath(root, epub_folder)
                    book_output_folder = os.path.join(output_folder, relative_path, epub_name)
                else:
                    book_output_folder = os.path.join(output_folder, epub_name)
                
                os.makedirs(book_output_folder, exist_ok=True)
                
                markdown_filename = f"{epub_name}.md"
                markdown_path = os.path.join(book_output_folder, markdown_filename)
                
                # Extract images directly to the book output folder
                extracted_images = list(extract_images_from_epub(epub_path, book_output_folder))
                
                # Get cover image, saving it directly to the book output folder
                cover_image = get_cover_image(book, book_output_folder)
                
                # Get title
                title = book.get_metadata('DC', 'title')[0][0]
                
                markdown_content = f"# {title}\n\n"
                
                if cover_image:
                    markdown_content += f"![Cover]({cover_image})\n\n"
                
                for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    
                    # Clean HTML content
                    cleaned_html = clean_html_content(soup, extracted_images)

                    # Convert HTML to Markdown
                    h = html2text.HTML2Text()
                    h.body_width = 0
                    markdown_content += h.handle(cleaned_html)

                    # Add page ID
                    page_id = item.get_id()
                    markdown_content += f"\n\n<!-- Page ID: {page_id} -->\n\n"

                # Ensure all image links are in the correct format
                markdown_content = re.sub(r'!\[(.*?)\]\((.*?)\)', lambda m: f'![{m.group(1)}]({m.group(2)})', markdown_content)

                # Save the Markdown content
                with open(markdown_path, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)

                print(f"Converted {filename} to Markdown: {markdown_path}")

    print("Conversion completed.")

def main():
    parser = argparse.ArgumentParser(description="Convert EPUB files to Markdown format.")
    parser.add_argument("epub_folder", help="Path to the folder containing EPUB files")
    parser.add_argument("output_folder", help="Path to the output folder for Markdown files")
    parser.add_argument("--preserve-structure", action="store_true", help="Preserve the folder structure of the EPUB files")
    args = parser.parse_args()

    epub_to_markdown(args.epub_folder, args.output_folder, args.preserve_structure)

if __name__ == "__main__":
    main()