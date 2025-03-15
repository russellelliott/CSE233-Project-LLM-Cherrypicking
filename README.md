Best of Both Worlds

Give the two directories, produces a 3rd which cherrypicks from the two sources, producing a result with the lowest number of API errors

first run `cherrypicking.py` to get the "best of both worlds" data

then run these files to get the neccesary graphs and data
`evaluation_graph_only.py` makes the graph for a given set
`evaluation_json_only.py` makes the json file for a given set

after that, run `success_rate.py` to produce success rate graps for the two experiments as well as a graph comparing the success rates of them

`distance.py` gets the manhattan distance and euclidean distance of the outputs from a directory
given the distance JSONs generated in distance.py, produces graphs showing the overall distance between generated and database code from the LLMs (distance_results)
`distance_graph.py` does a similar thing, but stacks them to access overall performance

`cosine_analysis.py` analyzes the cosine file and outputs the average of it across the results of the llms that succeeded

`success_rate.py` generates:
1. success rates for LLMs of vanila prompts (no added context)
    1. break down by LLM
    2. Overall performance of each LLM
2. success rates for LLMs of prompts with context
    1. break down by LLM
    2. Overall performance of each LLM
3. Comparison graph of overall performance of LLMs between context-free and context runs.