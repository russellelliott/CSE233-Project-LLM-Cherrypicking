import json
import os
import matplotlib.pyplot as plt
import numpy as np


def analyze_grouped_data(file_list):
    """
    Analyzes grouped data JSON files to determine LLM performance, including prompt-level success.

    Args:
        file_list (list): A list of filepaths to grouped data JSON files.

    Returns:
        tuple: (llm_performance, overall_success_rate, total_prompts, prompt_category_data)
               - llm_performance: LLM performance dictionary.
               - overall_success_rate: Overall success rate across all prompts.
               - total_prompts: Total number of unique prompts.
               - prompt_category_data: Dictionary of prompt-level success counts (organized by category).
    """

    llm_performance = {}
    total_prompts = 0
    prompt_category_data = {}  # Store success/rejection/api error counts for each prompt category

    # Iterate through the provided file list
    for filepath in file_list:
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            total_prompts += len(data)  # calculating the number of unique prompts used based on number of files and their content

            for index, llm_data in data.items():
                # Extract the category from the index (e.g., "1_1" -> "1")
                category = index.split('_')[0]
                if category not in prompt_category_data:
                    prompt_category_data[category] = {}

                for llm, counts in llm_data.items():
                    if llm not in llm_performance:
                        llm_performance[llm] = {"success": 0, "rejection": 0, "api_error": 0}

                    llm_performance[llm]["success"] += counts["success"]
                    llm_performance[llm]["rejection"] += counts["rejection"]
                    llm_performance[llm]["api_error"] += counts["api_error"]

                    # Add LLM-specific counts to the category data
                    if llm not in prompt_category_data[category]:
                        prompt_category_data[category][llm] = {"success": 0, "rejection": 0, "api_error": 0}
                    prompt_category_data[category][llm]["success"] += counts["success"]
                    prompt_category_data[category][llm]["rejection"] += counts["rejection"]
                    prompt_category_data[category][llm]["api_error"] += counts["api_error"]

        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error processing {filepath}: {e}")

    # Calculate overall success rate
    overall_success = sum(counts["success"] for counts in llm_performance.values())
    total_calls = sum(sum(counts.values()) for counts in llm_performance.values())
    overall_success_rate = (overall_success / total_calls) * 100 if total_calls > 0 else 0

    return llm_performance, overall_success_rate, total_prompts, prompt_category_data


def calculate_success_rate(llm_data):
    """Calculates the success rate for an LLM within a category."""
    total_calls = llm_data["success"] + llm_data["rejection"] + llm_data["api_error"]
    return (llm_data["success"] / total_calls) * 100 if total_calls > 0 else 0


def create_overall_performance_bar_graph(json_file, output_dir, file_prefix):
    """
    Creates a bar graph of overall LLM performance based on a JSON file.

    Args:
        json_file (str): Path to the JSON file containing the overall LLM performance data.
        output_dir (str): Directory to save the graph.
        file_prefix (str): Prefix for the output filename (to distinguish graphs).
    """

    with open(json_file, "r") as f:
        data = json.load(f)

    llms = list(data.keys())[:-2]  # Exclude the last two elements ("total_prompts", "overall_success_rate")
    success_rates = [data[llm]["success_rate"] for llm in llms]

    plt.figure(figsize=(12, 6))  # Adjust figure size if needed
    plt.bar(llms, success_rates, color='skyblue')
    plt.xlabel("LLMs")
    plt.ylabel("Success Rate (%)")
    plt.title("Overall LLM Performance")
    plt.ylim(0, 100)  # Set y-axis limit to 100%
    plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for better readability
    plt.tight_layout()  # Adjust layout to prevent labels from overlapping
    plt.savefig(os.path.join(output_dir, f"{file_prefix}_overall_llm_performance_bar_graph.png"))  # Save the graph as an image
    plt.close()


def create_prompt_category_bar_graph(json_file, output_dir, file_prefix):
    """
    Creates a bar graph of prompt category success rates for each LLM based on a JSON file.

    Args:
        json_file (str): Path to the JSON file containing the category performance data.
        output_dir (str): Directory to save the graph.
        file_prefix (str): Prefix for the output filename (to distinguish graphs).
    """
    with open(json_file, "r") as f:
        data = json.load(f)

    # Extract data for plotting
    categories = list(data.keys())[:-2]  # Exclude total_prompts and overall_success_rate
    llms = list(data[categories[0]].keys())  # Get the LLMs from the first category
    n_llms = len(llms)

    bar_width = 0.8 / n_llms
    group_width = 1

    plt.figure(figsize=(16, 8))
    index_positions = range(len(categories))

    # Plot each LLM's success rates
    for i, llm in enumerate(llms):
        x = [pos + (i - n_llms / 2 + bar_width/2) * bar_width for pos in index_positions]  #shift positions for each LLM
        success_rates = [data[category][llm]["success_rate"] for category in categories]
        plt.bar(x, success_rates, bar_width, label=llm)

    # Customize the plot
    plt.xlabel("Prompt Category", fontsize=14)
    plt.ylabel("Success Rate (%)", fontsize=14)
    plt.title("LLM Performance by Prompt Category")
    plt.xticks(index_positions, categories, rotation=45, ha="right")
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, f"{file_prefix}_prompt_category_performance_bar_graph.png"))
    plt.close()


def create_comparison_bar_graph(file1, file2, output_dir, filename="comparison_graph.png"):
    """
    Creates a bar graph comparing the overall LLM success rates from two JSON files.

    Args:
        file1 (str): Path to the first JSON file.
        file2 (str): Path to the second JSON file.
        output_dir (str): Directory to save the graph.
        filename (str): Name of the output image file.
    """

    with open(file1, "r") as f:
        data1 = json.load(f)
    with open(file2, "r") as f:
        data2 = json.load(f)

    llms1 = list(data1.keys())[:-2]
    llms2 = list(data2.keys())[:-2]

    # Ensure both files have the same LLMs (or handle missing LLMs gracefully)
    all_llms = sorted(list(set(llms1 + llms2)))

    success_rates1 = [data1.get(llm, {}).get("success_rate", 0) for llm in all_llms]  # Use get to handle missing keys
    success_rates2 = [data2.get(llm, {}).get("success_rate", 0) for llm in all_llms]  # Use get to handle missing keys

    x = np.arange(len(all_llms))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(14, 7))
    rects1 = ax.bar(x - width/2, success_rates1, width, label=os.path.basename(file1).replace("_overall_llm_performance.json", "").replace("_", " "))
    rects2 = ax.bar(x + width/2, success_rates2, width, label=os.path.basename(file2).replace("_overall_llm_performance.json", "").replace("_", " "))

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Success Rate (%)')
    ax.set_xlabel('LLMs')
    ax.set_title('LLM Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(all_llms, rotation=45, ha="right")
    ax.set_ylim(0, 100)
    ax.legend()

    fig.tight_layout()

    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

# removed prompt category comparison graph
# def create_comparison_prompt_category_graph(file1, file2, output_dir, filename="comparison_prompt_category_graph.png"):
#     """
#     Creates a grouped bar graph comparing prompt category success rates from two JSON files.
#     """
#     with open(file1, "r") as f:
#         data1 = json.load(f)
#     with open(file2, "r") as f:
#         data2 = json.load(f)

#     # Extract categories and LLMs
#     categories1 = list(data1.keys())[:-2]
#     categories2 = list(data2.keys())[:-2]
#     all_categories = sorted(list(set(categories1 + categories2)))  # Combine and sort categories

#     # Extract LLMs from the first category (assuming consistent LLMs across categories)
#     llms1 = list(data1.get(categories1[0], {}).keys()) if categories1 else []
#     llms2 = list(data2.get(categories2[0], {}).keys()) if categories2 else []
#     all_llms = sorted(list(set(llms1 + llms2)))

#     n_llms = len(all_llms)
#     bar_width = 0.8 / (2 * n_llms)  # Reduced bar width to accommodate both files
#     group_width = 1

#     plt.figure(figsize=(18, 8))  # Adjusted figure size for better readability
#     index_positions = np.arange(len(all_categories))

#     # Iterate through LLMs and plot bars for both files
#     for i, llm in enumerate(all_llms):
#         # Calculate x positions for the bars for file1
#         x1 = [pos + (i - n_llms / 2) * 2 * bar_width - bar_width/2 for pos in index_positions]
#         success_rates1 = [data1.get(category, {}).get(llm, {}).get("success_rate", 0) for category in all_categories]
#         plt.bar(x1, success_rates1, bar_width, label=f"{os.path.basename(file1).replace('_prompt_category_analysis.json', '').replace('_', ' ')} - {llm}")

#         # Calculate x positions for the bars for file2
#         x2 = [pos + (i - n_llms / 2) * 2 * bar_width + bar_width/2 for pos in index_positions]
#         success_rates2 = [data2.get(category, {}).get(llm, {}).get("success_rate", 0) for category in all_categories]
#         plt.bar(x2, success_rates2, bar_width, label=f"{os.path.basename(file2).replace('_prompt_category_analysis.json', '').replace('_', ' ')} - {llm}")

#     # Customize the plot
#     plt.xlabel("Prompt Category", fontsize=14)
#     plt.ylabel("Success Rate (%)", fontsize=14)
#     plt.title("LLM Performance by Prompt Category Comparison", fontsize=16)
#     plt.xticks(index_positions, all_categories, rotation=45, ha="right")
#     plt.ylim(0, 100)
#     plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Place legend outside the plot
#     plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to accommodate legend

#     plt.savefig(os.path.join(output_dir, filename))
#     plt.close()


def main():
    """
    Main function to perform the analysis, output JSON files, and create the bar graphs.
    """

    # Create the output directory if it doesn't exist
    output_dir = "success_rate"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define the file list and prefixes
    file_info = [
        {"file": "llm_performance/grouped_data_best_of_both_worlds.json", "prefix": "best_of_both_worlds"},
        {"file": "llm_performance/grouped_data_2025-03-08_09-55.json", "prefix": "date_2025_03_08"} #make it something safe
    ]

    # Perform analysis and generate individual graphs for each file
    for item in file_info:
        file_path = item["file"]
        file_prefix = item["prefix"]

        # Perform analysis for the current file
        llm_performance, overall_success_rate, total_prompts, prompt_category_data = analyze_grouped_data([file_path])

        # Prepare data for overall performance JSON output
        overall_json_output = {}
        ranked_llms = sorted(
            llm_performance.items(),
            key=lambda item: calculate_success_rate(item[1]),
            reverse=True
        )

        for rank, (llm, metrics) in enumerate(ranked_llms, 1):
            success_rate = calculate_success_rate(metrics)
            # Use .get() to safely access potential missing keys
            overall_json_output[llm] = {
                "successes": metrics.get("success", 0),
                "rejections": metrics.get("rejection", 0),  # Use .get() instead of direct access
                "api_errors": metrics.get("api_error", 0),  # Use .get() instead of direct access
                "success_rate": success_rate
            }

        overall_json_output["total_prompts"] = total_prompts
        overall_json_output["overall_success_rate"] = overall_success_rate

        # Output overall performance to JSON file
        overall_output_file = os.path.join(output_dir, f"{file_prefix}_overall_llm_performance.json")
        with open(overall_output_file, "w") as f:
            json.dump(overall_json_output, f, indent=4)
        print(f"\nOverall LLM performance saved to {overall_output_file}")

        # Prepare data for category-level performance JSON output
        category_json_output = {}
        for category, category_data in prompt_category_data.items():
            category_json_output[category] = {}
            for llm, metrics in category_data.items():
                success_rate = calculate_success_rate(metrics)
                category_json_output[category][llm] = {"success_rate": success_rate}
        category_json_output["total_prompts"] = total_prompts
        category_json_output["overall_success_rate"] = overall_success_rate

        # Output category-level performance to JSON file
        category_output_file = os.path.join(output_dir, f"{file_prefix}_prompt_category_analysis.json")
        with open(category_output_file, "w") as f:
            json.dump(category_json_output, f, indent=4)
        print(f"Category-level analysis saved to {category_output_file}")

        # Create the overall performance bar graph
        create_overall_performance_bar_graph(overall_output_file, output_dir, file_prefix)

        # Create the prompt category performance bar graph
        create_prompt_category_bar_graph(category_output_file, output_dir, file_prefix)

    # Generate comparison graphs
    overall_files = [os.path.join(output_dir, f"{item['prefix']}_overall_llm_performance.json") for item in file_info]
    # removed prompt category files in comparison
    # category_files = [os.path.join(output_dir, f"{item['prefix']}_prompt_category_analysis.json") for item in file_info]

    if len(overall_files) == 2:
        create_comparison_bar_graph(overall_files[0], overall_files[1], output_dir)
        # removed prompt category graph
        # create_comparison_prompt_category_graph(category_files[0], category_files[1], output_dir)
    else:
        print("Warning:  Need exactly two files to make comparisons. Skipping Comparison Graphs")


if __name__ == "__main__":
    main()