import argparse
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from jinja2 import Environment, FileSystemLoader
from tqdm import tqdm

from src.llms.openai_client import OpenAIClient

logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the instruction synthesis script."""
    parser = argparse.ArgumentParser(
        description="Process programming questions using GPT-4o."
    )
    parser.add_argument(
        "--input_path", required=True, help="Path to the input JSON file."
    )
    parser.add_argument(
        "--opt_evo", action="store_true", help="Enable optimization evolution mode."
    )
    parser.add_argument(
        "--output_dir", required=True, help="Directory to store output files."
    )
    parser.add_argument(
        "--num_threads", type=int, default=4, help="Number of threads to use (default: 4)."
    )
    parser.add_argument(
        "--num_responses", type=int, default=3, help="Number of responses to generate per item (default: 3)."
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Temperature for the model response (default: 1.0)."
    )
    parser.add_argument(
        "--max_tokens", type=int, default=2048, help="Maximum number of tokens in the response (default: 2048)."
    )
    parser.add_argument(
        "--model_name", type=str, default="gpt-4o", help="Model to use (default: gpt-4o)."
    )
    return parser.parse_args()


def extract_data(input_text: str) -> str:
    """
    Extract the programming question from the model's response.

    Args:
        input_text (str): The raw response from the model.

    Returns:
        str: The extracted programming question.
    """
    return input_text.strip().split(" [Programming Question]:")[-1].strip()


def load_json(file_path: str) -> Any:
    """
    Load a JSON file from the specified path.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        Any: The loaded JSON data.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(data: Any, file_path: str) -> None:
    """
    Write data to a file in JSON format.

    Args:
        data (Any): Data to write.
        file_path (str): Path to the output file.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def dump_jsonl(data: Any, file_path: str) -> None:
    """
    Write data to a file in JSON format (one object per file).

    Args:
        data (Any): Data to write.
        file_path (str): Path to the output file.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def process_item(client: OpenAIClient, obj: Dict[str, Any], config: Dict[str, Any]) -> Optional[str]:
    """
    Process a single item by generating responses using the LLM client and saving the results.

    Args:
        client (OpenAIClient): The LLM client.
        obj (Dict[str, Any]): The input object.
        config (Dict[str, Any]): Configuration for prompt and output.

    Returns:
        Optional[str]: The ID of the processed object, or None if skipped.
    """
    if not isinstance(obj["id"], str):
        obj["id"] = str(obj["id"])
    ids = obj["id"].split("_")
    content = obj["content"]
    template = config["template"]
    num_responses = config["num_responses"]
    output_dir = config["output_dir"]

    if len(ids) == 1:
        try:
            chosen_prompt = template.render(example=content.strip())
        except Exception as e:
            logger.error("Error rendering prompt for item %s: %s", obj['id'], e)
            return None
    else:
        comp_s = f"{obj.get('self complexity score'):.2f}"
        div_s = f"{obj.get('self diversity score') * 10:.2f}"
        try:
            chosen_prompt = template.render(
                example=content.strip(), comp_s=comp_s, div_s=div_s
            )
        except Exception as e:
            logger.error("Error rendering prompt for item %s: %s", obj['id'], e)
            return None

    for i in range(num_responses):
        output_path = os.path.join(output_dir, f"{obj['id']}_{i}.jsonl")
        if os.path.exists(output_path):
            logger.info("Skipping %s as it already exists.", output_path)
            return None

        try:
            response = client.request(
                prompt=chosen_prompt,
                temperature=config["temperature"],
                max_tokens=config["max_tokens"],
            )
            response = extract_data(response)
        except Exception as e:
            logger.error("Error processing item %s: %s", obj['id'], e)
            continue

        if len(response.split()) > 20:
            output = {
                "id": f"{obj['id']}_{i}",
                "content": response,
            }
            if len(ids) > 1:
                output["parent complexity score"] = obj.get("self complexity score")
                output["parent diversity score"] = obj.get("self diversity score")
            else:
                output["parent complexity score"] = "0"
                output["parent diversity score"] = "0"
            dump_jsonl(output, output_path)
    return obj['id']


def collect_results(output_dir: str) -> List[Dict[str, Any]]:
    """
    Collect and return all valid result objects from .jsonl files in the specified output directory.

    Args:
        output_dir (str): The directory containing .jsonl result files.

    Returns:
        List[Dict[str, Any]]: A list of result objects with non-empty 'content' fields.
    """
    results = []
    for filename in os.listdir(output_dir):
        if not filename.endswith(".jsonl"):
            continue
        file_path = os.path.join(output_dir, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict) and obj.get("content"):
                results.append(obj)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load %s: %s", file_path, e)
            continue
    return results


def main() -> None:
    """Main entry point for the instruction synthesis script."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    args = parse_arguments()

    data = load_json(args.input_path)
    client = OpenAIClient(model=args.model_name)

    # Load Jinja2 template
    prompts_dir = os.path.join(os.path.dirname(__file__), 'prompts')
    env = Environment(loader=FileSystemLoader(prompts_dir))
    template_name = 'optimization_driven_synthesis.jinja2' if args.opt_evo else 'direct_synthesis.jinja2'
    template = env.get_template(template_name)

    os.makedirs(args.output_dir, exist_ok=True)

    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = []
        for obj in data:
            config = {
                "template": template,
                "num_responses": args.num_responses,
                "output_dir": args.output_dir,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
            }
            future = executor.submit(process_item, client, obj, config)
            futures.append(future)

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing items"):
            result = future.result()
            if result:
                logger.info("Successfully processed item %s.", result)

    results = collect_results(args.output_dir)
    output_file = os.path.join(args.output_dir, f"all_questions_{len(results)}.json")
    dump_json(results, output_file)
    logger.info("All results written to %s", output_file)


if __name__ == "__main__":
    main()
