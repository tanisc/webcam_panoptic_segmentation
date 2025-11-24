import json
import os
import argparse
import sys


def sort_coco_json_categories(input_json_path, output_json_path):
    """
    Reads a COCO JSON file, sorts its categories alphabetically,
    updates the category IDs in the annotations, and saves the new file.

    Args:
        input_json_path (str): The path to the original COCO JSON file.
        output_json_path (str): The path where the sorted JSON file will be saved.
    """
    try:
        # Step 1: Read the original JSON file
        print(f"Reading data from: {input_json_path}")
        with open(input_json_path, 'r') as f:
            data = json.load(f)

        # Step 2: Create a mapping from the original category ID to its name
        original_categories = data['categories']
        old_id_to_name = {cat['id']: cat['name'] for cat in original_categories}
        
        print("Original categories found.")

        # Step 3: Sort the categories list alphabetically by name
        sorted_categories = sorted(original_categories, key=lambda x: x['name'])
        
        print("Categories sorted alphabetically.")

        # Step 4: Create a new mapping from category name to the new ID
        # and update the category objects with their new 1-based IDs
        name_to_new_id = {}
        for i, category in enumerate(sorted_categories):
            new_id = i + 1  # COCO category IDs are typically 1-based
            name_to_new_id[category['name']] = new_id
            category['id'] = new_id
        
        print("Created new 1-based category IDs.")

        # Step 5: Create a translation map from the old ID to the new ID
        old_id_to_new_id = {
            old_id: name_to_new_id[name]
            for old_id, name in old_id_to_name.items()
        }
        
        # Step 6: Iterate through all annotations and update their 'category_id'
        print("Updating category IDs in annotations...")
        for annotation in data['annotations']:
            old_category_id = annotation['category_id']
            if old_category_id in old_id_to_new_id:
                annotation['category_id'] = old_id_to_new_id[old_category_id]
            else:
                print(f"Warning: Annotation with old category_id {old_category_id} found, but this ID was not in the categories list.")

        # Step 7: Replace the old categories list with the new sorted one
        data['categories'] = sorted_categories

        # Step 8: Write the updated data to the new JSON file
        print(f"Saving sorted data to: {output_json_path}")
        with open(output_json_path, 'w') as f:
            json.dump(data, f, indent=4)
        
        print("\nSuccessfully sorted the COCO file!")

    except FileNotFoundError:
        print(f"Error: The file '{input_json_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# --- HOW TO USE ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sort categories in a COCO JSON file.")
    
    # Argument for the input file (required)
    parser.add_argument(
        'input_file', 
        default="snow_4cam_train_makesense.json",
        help="Path to the original COCO JSON file."
    )
    
    # Argument for the output file (optional)
    parser.add_argument(
        '-o', '--output', 
        help="Path for the sorted output file. Defaults to <input>_sorted.json",
        default=None
    )

    args = parser.parse_args()

    input_file = args.input_file

    # Determine output filename
    if args.output:
        output_file = args.output
    else:
        # Auto-generate name: filename.json -> filename_sorted.json
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_sorted{ext}"

    # Validate input existence
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at '{os.path.abspath(input_file)}'.")
        sys.exit(1)
    
    sort_coco_json_categories(input_file, output_file)
