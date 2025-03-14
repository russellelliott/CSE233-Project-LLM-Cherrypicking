import json
import os
import matplotlib.pyplot as plt


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


def create_overall_performance_bar_graph(json_file):
    """
    Creates a bar graph of overall LLM performance based on a JSON file.

    Args:
        json_file (str): Path to the JSON file containing the overall LLM performance data.
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
    plt.savefig("overall_llm_performance_bar_graph.png")  # Save the graph as an image
    plt.show()


def main():
    """
    Main function to perform the analysis, output JSON files, and create the bar graph.
    """

    file_list = [
        "llm_performance/grouped_data_best_of_both_worlds.json",
        "llm_performance/grouped_data_2025-03-08_09-55.json"
    ]

    llm_performance, overall_success_rate, total_prompts, prompt_category_data = analyze_grouped_data(file_list)

    # Prepare data for overall performance JSON output
    overall_json_output = {}
    ranked_llms = sorted(
        llm_performance.items(),
        key=lambda item: calculate_success_rate(item[1]),
        reverse=True
    )

    for rank, (llm, metrics) in enumerate(ranked_llms, 1):
        success_rate = calculate_success_rate(metrics)
        overall_json_output[llm] = {
            "successes": metrics["success"],
            "rejections": metrics["rejection"],
            "api_errors": metrics["api_error"],
            "success_rate": success_rate
        }

    overall_json_output["total_prompts"] = total_prompts
    overall_json_output["overall_success_rate"] = overall_success_rate

    # Output overall performance to JSON file
    overall_output_file = "overall_llm_performance.json"
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
    category_output_file = "prompt_category_analysis.json"
    with open(category_output_file, "w") as f:
        json.dump(category_json_output, f, indent=4)
    print(f"Category-level analysis saved to {category_output_file}")

    # Create the overall performance bar graph
    create_overall_performance_bar_graph(overall_output_file)


if __name__ == "__main__":
    main()