import json
import os

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


def main():
    """
    Main function to perform the analysis, print the results, and output prompt-level data to a text file.
    """

    file_list = [
        "llm_performance/grouped_data_best_of_both_worlds.json",
        "llm_performance/grouped_data_2025-03-08_09-55.json"
    ]

    llm_performance, overall_success_rate, total_prompts, prompt_category_data = analyze_grouped_data(file_list)

    output_file = "prompt_category_analysis.txt"
    with open(output_file, "w") as f:
        f.write("LLM Performance Analysis by Prompt Category:\n\n")

        # Iterate through each prompt category
        for category, category_data in prompt_category_data.items():
            f.write(f"Category: {category}\n")

            # Calculate and sort LLMs by success rate within this category
            ranked_llms = sorted(
                category_data.items(),
                key=lambda item: calculate_success_rate(item[1]),
                reverse=True
            )

            # Output the ranked LLM performance for the category
            for rank, (llm, metrics) in enumerate(ranked_llms, 1):
                success_rate = calculate_success_rate(metrics)
                f.write(f"{rank}. {llm}:\n")
                f.write(f"   Successes: {metrics['success']}\n")
                f.write(f"   Rejections: {metrics['rejection']}\n")
                f.write(f"   API Errors: {metrics['api_error']}\n")
                f.write(f"   Success Rate: {success_rate:.2f}%\n")

            f.write("\n")  # Add a separator between categories

        # Overall summary
        f.write("Overall LLM Performance:\n")
        ranked_llms_overall = sorted(
            llm_performance.items(),
            key=lambda item: (item[1]["success"] + item[1]["rejection"] + item[1]["api_error"]),  # Sort by total calls
            reverse=True
        )

        # Output overall metrics.
        total_prompts = len(prompt_category_data)
        f.write(f"\nTotal number of unique prompts: {total_prompts}\n")
        f.write(f"Overall Success Rate: {overall_success_rate:.2f}%\n")


    print(f"\nPrompt-level analysis saved to {output_file}")


if __name__ == "__main__":
    main()