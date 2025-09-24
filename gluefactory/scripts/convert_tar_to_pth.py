import torch
import os
import argparse
from collections import OrderedDict

def convert_tar_to_pth(input_tar_path, output_pth_path):
    """
    Loads a PyTorch model checkpoint from a .tar file, removes specific
    prefixes often added by DistributedDataParallel (DDP) or other
    wrappers, and saves the cleaned state dictionary to a new .pth file.

    Args:
        input_tar_path (str): The path to the input .tar checkpoint file.
        output_pth_path (str): The path to save the output .pth file.
    """
    # Check if the input file exists
    if not os.path.exists(input_tar_path):
        print(f"Error: Input file '{input_tar_path}' not found.")
        return

    print(f"Loading checkpoint from: {input_tar_path}")
    checkpoint = torch.load(input_tar_path)
    
    # Print all the keys found in the checkpoint to help with debugging
    print(f"Checkpoint keys found: {checkpoint.keys()}")
    
    # Extract the state dictionary from the checkpoint.
    # The key might be 'model_state_dict', 'state_dict', or just 'model'.
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        print("Error: Could not find a standard state dictionary key.")
        print("Please manually specify the correct key from the list above.")
        return

    new_state_dict = OrderedDict()
    
    # Iterate through the keys and remove the specified prefixes
    print("Removing 'matcher.' and 'extractor.' prefixes from state dictionary keys...")
    for k, v in state_dict.items():
        print(f"key: {k}")
        if k.startswith('matcher.'):
            name = k[8:]  # Remove 'matcher.'
            new_state_dict[name] = v
        elif k.startswith('extractor.'):
            continue
            #name = k[10:]  # Remove 'extractor.'
            #new_state_dict[name] = v
        else:
            # If no prefix is present, use the key as is
            print(f"no prefix key: {k}")
            new_state_dict[k] = v

    # Save the new state dictionary to the .pth file
    torch.save(new_state_dict, output_pth_path)
    print(f"Successfully saved cleaned state dictionary to: {output_pth_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert a PyTorch .tar checkpoint with DDP prefixes to a .pth state dict.")
    parser.add_argument("input_tar", help="Path to the input .tar file.")
    parser.add_argument("--output_pth", help="Optional: Path for the output .pth file. If not provided, a .pth file with the same name will be created.", default=None)

    args = parser.parse_args()

    # If the output path is not provided, generate a default one
    if args.output_pth is None:
        file_name = os.path.splitext(os.path.basename(args.input_tar))[0]
        output_dir = os.path.dirname(args.input_tar)
        output_pth_path = os.path.join(output_dir, f"{file_name}.pth")
    else:
        output_pth_path = args.output_pth
    
    convert_tar_to_pth(args.input_tar, output_pth_path)

if __name__ == '__main__':
    main()

