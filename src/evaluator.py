from langsmith import Client
from langsmith.evaluation import run_evaluator
from langsmith.schemas import Example, Run
from langchain.smith import RunEvalConfig

from src.agent import app

# Initialize the LangSmith client
client = Client()

# Define a simple custom evaluator that checks if the output is not empty
@run_evaluator
def must_not_be_empty(run: Run, example: Example) -> dict:
    """
    A simple evaluator that checks if the agent's output is not empty.
    """
    if not run.outputs or not run.outputs.get("answer"):
        return {"key": "not_empty", "score": 0, "comment": "Output is empty."}
    return {"key": "not_empty", "score": 1}

def evaluate_agent():
    """
    Runs a programmatic evaluation of the agent on a small dataset.
    """
    print("--- Starting Programmatic Evaluation ---")
    
    # Define a dataset of questions for evaluation
    dataset_name = "AgentTestDataset"
    
    # Create the dataset if it doesn't exist
    if not client.has_dataset(dataset_name=dataset_name):
        client.create_dataset(dataset_name=dataset_name)
        
        # Add examples to the dataset
        client.create_examples(
            inputs=[
                {"question": "Delhi"},
                {"question": "tell me about avdeep's experience? in 2 lines"},
                {"question": "What is the capital of France?"} # A question not in the PDF
            ],
            dataset_name=dataset_name,
        )
        print(f"Dataset '{dataset_name}' created with 3 examples.")
    else:
        print(f"Using existing dataset '{dataset_name}'.")
        
    evaluation_result = client.run_on_dataset(
        dataset_name=dataset_name,
        llm_or_chain_factory=app,
        evaluation=RunEvalConfig(
            custom_evaluators=[must_not_be_empty]  # List of custom evaluator functions
        ),
        project_name=None,
        concurrency_level=1,
    )
    
    print("--- Evaluation Finished ---")
    print("You can view the results in your LangSmith project.")

if __name__ == "__main__":
    evaluate_agent()