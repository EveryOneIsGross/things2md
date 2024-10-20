import fitz
import re
import argparse
import os

def clean_svg(svg_content):
    # Remove XML declaration
    svg_content = re.sub(r'<\?xml[^>]+\?>\n', '', svg_content)
    
    # Remove comments
    svg_content = re.sub(r'<!--.*?-->\n', '', svg_content, flags=re.DOTALL)
    
    # Remove empty defs
    svg_content = re.sub(r'<defs>\s*</defs>\n', '', svg_content)
    
    return svg_content.strip()

def pdf_to_svg(pdf_path, svg_path):
    doc = fitz.open(pdf_path)
    
    all_svg_content = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        svg_content = page.get_svg_image(matrix=fitz.Identity)
        
        # Clean the SVG content
        svg_content = clean_svg(svg_content)
        
        # For multi-page PDFs, we'll only keep one SVG tag opener and closer
        if page_num == 0:
            all_svg_content.append(svg_content)
        else:
            # Remove the SVG opening and closing tags for pages after the first
            svg_content = re.sub(r'<svg[^>]*>', '', svg_content)
            svg_content = re.sub(r'</svg>', '', svg_content)
            all_svg_content.append(svg_content.strip())
    
    # Combine all pages
    final_svg = '\n'.join(all_svg_content)
    
    # Save the SVG file
    with open(svg_path, 'w', encoding='utf-8') as f:
        f.write(final_svg)

    print(f"SVG file saved to {svg_path}")

def process_files(input_path, output_dir):
    if os.path.isfile(input_path):
        # Process single file
        filename = os.path.basename(input_path)
        name, _ = os.path.splitext(filename)
        output_path = os.path.join(output_dir, f"{name}.svg")
        pdf_to_svg(input_path, output_path)
    elif os.path.isdir(input_path):
        # Process all PDF files in the directory
        for filename in os.listdir(input_path):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(input_path, filename)
                name, _ = os.path.splitext(filename)
                output_path = os.path.join(output_dir, f"{name}.svg")
                pdf_to_svg(pdf_path, output_path)
    else:
        print(f"Error: {input_path} is not a valid file or directory")

def main():
    parser = argparse.ArgumentParser(description="Convert PDF files to SVG format.")
    parser.add_argument("input", help="Input PDF file or directory containing PDF files")
    parser.add_argument("-o", "--output", default=".", help="Output directory for SVG files (default: current directory)")
    args = parser.parse_args()

    input_path = args.input
    output_dir = args.output

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        process_files(input_path, output_dir)
        print("Conversion completed successfully.")
    except Exception as e:
        print(f"An error occurred during conversion: {e}")

if __name__ == "__main__":
    main()