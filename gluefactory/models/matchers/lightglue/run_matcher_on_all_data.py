import os
import re
import subprocess
import shutil
import argparse
from PIL import Image, ImageDraw, ImageFont


def run_matchers_in_patch_data_dirs(base_dir, output_dir):
    base_dir = os.path.abspath(base_dir)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    patch_dirs = [
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if d.endswith("patch_data") and os.path.isdir(os.path.join(base_dir, d))
    ]

    if not patch_dirs:
        print(f"No directories ending with 'patch_data' found in {base_dir}.")
        return

    all_results = []  # store (image_path, title)

    for patch_dir in patch_dirs:
        print(f"\n--- Processing {patch_dir} ---")
        os.chdir(patch_dir)

        # Run matcher script
        cmd = ["python3", "/usr/local/home/rsoussan/glue-factory/gluefactory/models/matchers/lightglue/run_matcher.py"]
        print(f"Running: {' '.join(cmd)} in {os.getcwd()}")
        subprocess.run(cmd, check=True)

        # Collect resulting .png files
        png_files = [f for f in os.listdir(".") if f.endswith(".png")]
        if not png_files:
            print(f"No PNG files generated in {patch_dir}.")
            os.chdir(base_dir)
            continue

        # Create results directory for this patch_data folder
        patch_name = os.path.basename(patch_dir)
        patch_output_dir = os.path.join(output_dir, f"{patch_name}_results")
        os.makedirs(patch_output_dir, exist_ok=True)

        # Move .png files to results directory
        for png_file in png_files:
            src = os.path.join(os.getcwd(), png_file)
            dst = os.path.join(patch_output_dir, png_file)
            shutil.move(src, dst)
            print(f"Moved {src} -> {dst}")

            title = f"{patch_name}: {png_file}"
            all_results.append((dst, title))

        os.chdir(base_dir)

    print("\n All patch_data directories processed successfully.")

    # --- Create combined PDF if requested ---
    if all_results:
        pdf_path = os.path.join(output_dir, "all_match_results.pdf")
        create_combined_pdf(all_results, pdf_path)

def natural_key(pair):
    """Generate a natural sort key from the image path."""
    path = pair[0]
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', path)]

def create_combined_pdf(image_title_pairs, output_pdf):
    """Combine PNGs into one PDF, labeling each page with the file and directory name."""
    pages = []
    # Sort in-place
    image_title_pairs.sort(key=natural_key)

    for img_path, title in image_title_pairs:
        img = Image.open(img_path).convert("RGB")

        # Draw title text at top of image
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 20)
        except:
            font = ImageFont.load_default()
        text_w, text_h = draw.textsize(title, font=font)

        # Add padding for title
        new_img = Image.new("RGB", (img.width, img.height + text_h + 10), (255, 255, 255))
        new_img.paste(img, (0, text_h + 10))
        draw = ImageDraw.Draw(new_img)
        draw.text(((img.width - text_w) // 2, 5), title, fill=(0, 0, 0), font=font)

        pages.append(new_img)

    print(f"\n Creating combined PDF: {output_pdf}")
    pages[0].save(output_pdf, save_all=True, append_images=pages[1:])
    print(f"PDF saved: {output_pdf}")


def main():
    parser = argparse.ArgumentParser(
        description="Run matcher on all *_patch_data dirs and optionally create a PDF of results."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=".",
        help="Base directory to search for *_patch_data folders (default: current dir).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Output directory to store results and PDFs (default: current dir).",
    )

    args = parser.parse_args()
    run_matchers_in_patch_data_dirs(args.base_dir, args.output_dir)


if __name__ == "__main__":
    main()

