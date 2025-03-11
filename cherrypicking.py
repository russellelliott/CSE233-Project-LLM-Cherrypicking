import os
import json
import asyncio
import aiofiles

# Configuration - Adjust these as needed
DIRECTORY_1 = "March 9 Context Experiment"  # Directory with potentially erroneous responses
DIRECTORY_2 = "March 11 Context Experiment"  # Directory with potentially better responses
DIRECTORY_3 = "best_of_both_worlds"  # The directory to store the combined results
LLM_LIST = ["llama3-8b-8192", "gemini-2.0-flash", "gpt-4o", "claude-3-5-sonnet-20241022", "deepseek-chat"]


async def load_json(file_path):
    """Loads JSON data from a file asynchronously."""
    try:
        async with aiofiles.open(file_path, 'r') as f:
            return json.loads(await f.read())
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading JSON from {file_path}: {e}")
        return None


async def process_file(filename, dir1_path, dir2_path, dir3_path):
    """Processes a single file, combining the best responses from dir1 and dir2."""

    file1_path = os.path.join(dir1_path, filename)
    file2_path = os.path.join(dir2_path, filename)
    file3_path = os.path.join(dir3_path, filename)

    data1 = await load_json(file1_path)
    data2 = await load_json(file2_path)

    if data1 is None and data2 is None:
        print(f"Skipping {filename} due to loading errors in both files.")
        return

    # If one file doesn't exist, just copy the other
    if data1 is None:
        print(f"Copying {filename} from directory 2 to directory 3 as directory 1 is missing.")
        data3 = data2
    elif data2 is None:
        print(f"Copying {filename} from directory 1 to directory 3 as directory 2 is missing.")
        data3 = data1
    else:
        data3 = await combine_responses(data1, data2) # Actual combining happens here

    if data3:  # ensure data3 is not None
        try:
            async with aiofiles.open(file3_path, "w") as outfile:
                await outfile.write(json.dumps(data3, indent=4))
            print(f"Successfully created/updated: {file3_path}")
        except IOError as e:
            print(f"Error writing to file: {e}")


async def combine_responses(data1, data2):
    """Combines the best responses from two JSON data lists."""
    if not isinstance(data1, list) or not isinstance(data2, list):
        print("Error: Data is not a list.  Skipping combination.")
        return None

    combined_data = []
    # Create a dictionary to quickly look up entries from data2 by index
    data2_dict = {entry.get("Index"): entry for entry in data2}

    for entry1 in data1:
        index = entry1.get("Index")
        entry2 = data2_dict.get(index)

        if entry2:
            combined_entry = await choose_best_response(entry1, entry2) # Choose best response
        else:
            combined_entry = entry1  # Use entry1 if no corresponding entry in data2
            print(f"Warning: Index {index} not found in data2, using data1's entry.")

        combined_data.append(combined_entry)

    # Add entries from data2 that are not in data1
    for entry2 in data2:
        index = entry2.get("Index")
        if index not in [entry.get("Index") for entry in data1]:
            combined_data.append(entry2)
            print(f"Adding entry from data2 with Index {index} as it's not present in data1.")


    return combined_data


async def choose_best_response(entry1, entry2):
    """Chooses the best response for each task and model, prioritizing non-error responses."""

    combined_entry = entry1.copy()  # Start with a copy of entry1
    for task_type in ["Summary", "Details"]:
        responses1 = entry1["Responses"][task_type]
        responses2 = entry2["Responses"][task_type]
        combined_responses = combined_entry["Responses"][task_type]

        for model_name in LLM_LIST:
            response1 = responses1.get(model_name, "")
            response2 = responses2.get(model_name, "")

            if "API Error" in response1 and "API Error" not in response2:
                combined_responses[model_name] = response2
                print(f"  Chose data2 for Index: {entry1['Index']}, Model: {model_name}, Task: {task_type}")
            elif "API Error" not in response1 and "API Error" in response2:
                # Keep response1 as it is not an error and response2 is
                print(f"  Chose data1 for Index: {entry1['Index']}, Model: {model_name}, Task: {task_type}")
                pass # keep response1 since it is good
            else:
                # If both have errors or both are good, default to data1 (you can change this logic)
                print(f"  Both files have same result for Index: {entry1['Index']}, Model: {model_name}, Task: {task_type}, choosing data1")
                pass # Do nothing, keeping entry1's response.

    return combined_entry


async def main():
    """Main function to orchestrate the process."""

    dir1_path = DIRECTORY_1
    dir2_path = DIRECTORY_2
    dir3_path = DIRECTORY_3

    # Ensure directories exist
    for path in [dir1_path, dir2_path]:
        if not os.path.isdir(path):
            print(f"Error: Directory not found: {path}")
            return

    # Create directory 3 if it doesn't exist
    if not os.path.exists(dir3_path):
        os.makedirs(dir3_path)
        print(f"Created directory: {dir3_path}")


    # Gather all the file processing tasks
    tasks = [
        process_file(filename, dir1_path, dir2_path, dir3_path)
        for filename in os.listdir(dir1_path)
        if "output_index" in filename and filename.endswith(".json") and os.path.exists(os.path.join(dir2_path, filename))
    ]

    # Run the tasks concurrently
    await asyncio.gather(*tasks)

    print("All files processed.")


if __name__ == "__main__":
    asyncio.run(main())