import os
import subprocess
import shutil

def run_matchers_in_patch_data_dirs():
    base_dir = os.getcwd()
    patch_dirs = [d for d in os.listdir(base_dir) if d.endswith("patch_data") and os.path.isdir(d)]

    if not patch_dirs:
        print("No directories ending with 'patch_data' found.")
        return

    for patch_dir in patch_dirs:
        print(f"\n--- Processing {patch_dir} ---")
        os.chdir(patch_dir)

        # Run matcher script
        cmd = ["python3", "/home/rsoussan/glue-factory/gluefactory/models/matchers/lightglue/run_matcher.py"]
        print(f"Running: {' '.join(cmd)} in {os.getcwd()}")
        subprocess.run(cmd, check=True)

        # Collect resulting .png files
        png_files = [f for f in os.listdir(".") if f.endswith(".png")]
        if not png_files:
            print(f"No PNG files generated in {patch_dir}.")
            os.chdir(base_dir)
            continue

        # Create output directory prefixed by patch_dir name
        output_dir = os.path.join(base_dir, f"{patch_dir}_results")
        os.makedirs(output_dir, exist_ok=True)

        # Move .png files to results directory
        for png_file in png_files:
            src = os.path.join(os.getcwd(), png_file)
            dst = os.path.join(output_dir, png_file)
            shutil.move(src, dst)
            print(f"Moved {src} -> {dst}")

        # Return to base directory before continuing
        os.chdir(base_dir)

    print("\n All patch_data directories processed successfully.")

def main():
    run_matchers_in_patch_data_dirs()

if __name__ == "__main__":
    main()

