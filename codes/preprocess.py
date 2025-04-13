import os

def preprocess_and_extract_smiles(input_filepath, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Output file path
    output_filepath = os.path.join(output_folder, os.path.basename(input_filepath))

    with open(input_filepath, "r") as infile, open(output_filepath, "w") as outfile:
        for line in infile:
            # Split by whitespace (handles both spaces and tabs) and take only the first part (SMILES)
            smiles = line.split()[0]
            outfile.write(smiles + "\n")  # Write only the SMILES to the new file

    print(f"Processed data saved to: {output_filepath}")

# Example usage
if __name__ == "__main__":
    input_filepath = "/home/satya/Desktop/BI/dataset/CCAB.smi"  # Update with actual dataset path
    output_folder = "/home/satya/Desktop/BI/processed_data/"
    
    preprocess_and_extract_smiles(input_filepath, output_folder)
