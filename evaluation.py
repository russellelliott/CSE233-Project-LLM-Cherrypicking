import json
import matplotlib.pyplot as plt
import numpy as np
import os
import re

def create_bar_chart(data, llm_list, output_filename="llm_performance.png"):
    """
    Generates a grouped and stacked bar chart showing the aggregate performance of different LLMs
    (success, rejection, api_error) for each top-level index (e.g., 1, 2, 3...).

    Args:
        data (dict):  The data containing the analysis results.
        llm_list (list): A list of the LLMs to include in the chart.
        output_filename (str): The name of the file to save the chart to.
    """

    # Group by top-level index
    grouped_data = {}
    for index, llm_data in data.items():
        top_level_index = int(index.split('_')[0])  # Extract the number before the underscore
        if top_level_index not in grouped_data:
            grouped_data[top_level_index] = {}
        for llm, counts in llm_data.items():
            if llm not in grouped_data[top_level_index]:
                grouped_data[top_level_index][llm] = {"success": 0, "rejection": 0, "api_error": 0}
            grouped_data[top_level_index][llm]["success"] += counts["success"]
            grouped_data[top_level_index][llm]["rejection"] += counts["rejection"]
            grouped_data[top_level_index][llm]["api_error"] += counts["api_error"]

    # Prepare data for plotting
    indices = sorted(grouped_data.keys())
    n_indices = len(indices)
    n_llms = len(llm_list)
    bar_width = 0.8 / n_llms
    group_width = 1

    index_positions = np.arange(n_indices) * group_width

    success_counts = {llm: [] for llm in llm_list}
    rejection_counts = {llm: [] for llm in llm_list}
    api_error_counts = {llm: [] for llm in llm_list}

    for top_level_index in indices:
        for llm in llm_list:
            if llm in grouped_data[top_level_index]:
                success_counts[llm].append(grouped_data[top_level_index][llm]["success"])
                rejection_counts[llm].append(grouped_data[top_level_index][llm]["rejection"])
                api_error_counts[llm].append(grouped_data[top_level_index][llm]["api_error"])
            else:
                success_counts[llm].append(0)
                rejection_counts[llm].append(0)
                api_error_counts[llm].append(0)

    # Plotting
    fig, ax = plt.subplots(figsize=(15, 8))

    success_color = '#55a868'
    rejection_color = '#c44e52'
    api_error_color = '#dd8452'

    for i, llm in enumerate(llm_list):
        x = index_positions + (i - n_llms / 2 + bar_width/2) * bar_width

        success_counts_np = np.array(success_counts[llm])
        rejection_counts_np = np.array(rejection_counts[llm])
        api_error_counts_np = np.array(api_error_counts[llm])

        ax.bar(x, success_counts_np, bar_width, label=f'Success' if i == 0 else None, color=success_color)
        ax.bar(x, rejection_counts_np, bar_width, bottom=success_counts_np, label=f'Rejection' if i == 0 else None, color=rejection_color)
        ax.bar(x, api_error_counts_np, bar_width, bottom=success_counts_np + rejection_counts_np, label=f'API Error' if i == 0 else None, color=api_error_color)

    # Customize the plot
    ax.set_xlabel('Top-Level Index', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    ax.set_title('LLM Performance by Top-Level Index', fontsize=16)
    ax.set_xticks(index_positions)
    ax.set_xticklabels(indices, rotation=45, ha="right")
    ax.set_ylim([0, 60])  # 30 prompts in a given category, 2 calls per LLM, 60 total calls per LLM

    ax.grid(axis='y', linestyle='--')
    ax.legend(loc='upper right', ncol=n_llms)

    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join("llm_performance", output_filename))
    plt.show()
    print(f"Chart saved to {os.path.join('llm_performance', output_filename)}")

def analyze_directories(selected_dirs, analysis_results_dir="analysis_results", output_prefix="llm_performance"):
    """
    Analyzes multiple directories, loading corresponding 'analysis_results.json' files
    and generates a bar chart *for each* directory, along with a JSON file summarizing
    the aggregated results.

    Args:
        selected_dirs (list): A list of directory paths to API responses.
        analysis_results_dir (str): The base directory where the analysis results are stored.
        output_prefix (str): The prefix for the output filenames.
    """

    # Create the "llm_performance" directory if it doesn't exist
    if not os.path.exists("llm_performance"):
        os.makedirs("llm_performance")

    for api_response_dir in selected_dirs: #iterate through each of the response files
        # Extract the date/context path name from the API response directory
        date_context_path = os.path.basename(api_response_dir)

        # Construct the full path to the 'analysis_results.json' file
        analysis_file_dir = os.path.join(analysis_results_dir, date_context_path)
        analysis_file_path = os.path.join(analysis_file_dir, "analysis_results.json")

        try:
            # Load the JSON data from the analysis results file
            with open(analysis_file_path, "r") as f:
                data = json.load(f)

            # Extract LLM list from the first entry
            llm_list = []
            for prompt_index, llm_data in data.items():
                llm_list.extend(llm_data.keys())
                break  # Only need to examine the first prompt index
            llm_list = list(set(llm_list))  # Remove duplicates

            # Group the data
            grouped_data = {}
            for index, llm_data in data.items():
                top_level_index = int(index.split('_')[0])  # Extract the number before the underscore
                if top_level_index not in grouped_data:
                    grouped_data[top_level_index] = {}
                for llm, counts in llm_data.items():
                    if llm not in grouped_data[top_level_index]:
                        grouped_data[top_level_index][llm] = {"success": 0, "rejection": 0, "api_error": 0}
                    grouped_data[top_level_index][llm]["success"] += counts["success"]
                    grouped_data[top_level_index][llm]["rejection"] += counts["rejection"]
                    grouped_data[top_level_index][llm]["api_error"] += counts["api_error"]

            # Save grouped data to JSON
            grouped_json_filename = f"grouped_data_{date_context_path}.json"
            output_path = os.path.join("llm_performance", grouped_json_filename)
            with open(output_path, "w") as outfile:
                json.dump(grouped_data, outfile, indent=4)
            print(f"Grouped JSON data saved to {output_path}")


            # Create the bar chart for the current analysis results file
            output_filename = f"{output_prefix}_{date_context_path}.png"
            create_bar_chart(data, llm_list, output_filename)

        except FileNotFoundError:
            print(f"Error: 'analysis_results.json' not found in directory: {analysis_file_path}")
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in: {analysis_file_path}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")


selected_dirs = [
    "March 9 Context Experiment", #original prompts
    "March 11 Context Experiment" #added jailbreak prompt 2
]

# Analyze the directories and generate charts
analyze_directories(selected_dirs)