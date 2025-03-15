import json
import matplotlib.pyplot as plt
import numpy as np
import os
import re

def load_cosine_distances(filepath):
    """Loads cosine distances from a JSON file."""
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading {filepath}: {e}")
        return None


def create_stacked_bar_chart(data, llm_list, graph_title, output_filename="cosine_distances_stacked.png"):
    """
    Generates a stacked bar chart of average cosine distances for Summary and Details prompts,
    grouped by LLM.  Assumes data is already loaded and formatted.
    """

    num_prompts = len(data["prompt_distances"])
    n_llms = len(llm_list)
    bar_width = 0.7
    index = np.arange(n_llms)

    fig, ax = plt.subplots(figsize=(12, 7))

    summary_averages = {}
    detail_averages = {}

    # Calculate average cosine distances for each LLM and prompt type
    for llm in llm_list:
        summary_total = 0.0
        detail_total = 0.0
        summary_count = 0
        detail_count = 0

        for item in data["prompt_distances"]:
            llm_distances = item["LLM_distances"].get(llm)
            if llm_distances:
                summary_distance = llm_distances.get("Summary", {}).get("cosine")
                detail_distance = llm_distances.get("Details", {}).get("cosine")

                if summary_distance is not None:
                    summary_total += summary_distance
                    summary_count += 1
                if detail_distance is not None:
                    detail_total += detail_distance
                    detail_count += 1

        # Calculate averages, handling potential division by zero.  Don't graph if zero.
        summary_averages[llm] = summary_total / summary_count if summary_count > 0 else 0
        detail_averages[llm] = detail_total / detail_count if detail_count > 0 else 0

    # Prepare data for plotting
    bottom = np.zeros(n_llms)
    summary_values = []
    detail_values = []

    for llm in llm_list:
      summary_values.append(summary_averages[llm])
      detail_values.append(detail_averages[llm])

    # Plotting the stacked bars
    ax.bar(index, summary_values, bar_width, label='Summary', color='skyblue', bottom=bottom)
    bottom = np.array(summary_values) # Convert the lists to numpy arrays to use for bototm
    ax.bar(index, detail_values, bar_width, label='Details', color='lightcoral', bottom=bottom)

    ax.set_xlabel('LLM', fontsize=12)
    ax.set_ylabel('Average Cosine Distance', fontsize=12)
    ax.set_title(graph_title, fontsize=14) # Graph title
    ax.set_xticks(index)
    ax.set_xticklabels(llm_list, rotation=45, ha="right")
    ax.legend()

    fig.tight_layout()
    plt.savefig(os.path.join("llm_performance", output_filename))
    plt.show()
    print(f"Stacked cosine distance chart saved to {os.path.join('llm_performance', output_filename)}")


def process_file(filepath, graph_title):
    """Processes a single JSON file to create a chart."""
    # Define the desired LLM order
    LLM_LIST = ["llama3-8b-8192", "gemini-2.0-flash", "gpt-4o", "claude-3-5-sonnet-20241022", "deepseek-chat"]

    data = load_cosine_distances(filepath)

    if data:
        # Extract filename without extension for the chart name
        filename_without_extension = os.path.splitext(os.path.basename(filepath))[0]
        output_filename = f"{filename_without_extension}_stacked.png"

        create_stacked_bar_chart(data, LLM_LIST, graph_title, output_filename)
    else:
        print(f"Could not load cosine distances from {filepath}. Check directory and file formats.")


def main():
    """Main function to process multiple files."""
    filepaths = {
        "distance_results/distance_results_2025-03-08_09-55.json": "Average Cosine Distance by LLM (Stacked) - Vanilla Prompts",
        "distance_results/distance_results_best_of_both_worlds.json": "Average Cosine Distance by LLM (Stacked) - Prompts With Context"
    }

    if not os.path.exists("llm_performance"):
        os.makedirs("llm_performance")

    # Create the 'distance_results' directory if it doesn't exist
    distance_results_dir = os.path.dirname(list(filepaths.keys())[0]) # Get the first filepath
    if not os.path.exists(distance_results_dir):
        os.makedirs(distance_results_dir)

    # Create dummy JSON files if they don't exist (for testing purposes)
    for filepath in filepaths:
        if not os.path.exists(filepath):
            print(f"Warning: File {filepath} not found. Creating a dummy file for testing.")
            dummy_data = {
                "prompt_distances": []  # Replace with your actual data structure for testing
            }
            with open(filepath, "w") as f:
                json.dump(dummy_data, f, indent=4)

    for filepath, graph_title in filepaths.items():
        process_file(filepath, graph_title)


if __name__ == "__main__":
    main()