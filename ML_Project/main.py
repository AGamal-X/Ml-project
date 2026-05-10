from src.common.document_builder import build_project_documents
from src.common.helpers import ensure_project_directories, save_json
from src.image.utils import run_image_pipeline
from src.numerical.utils import run_numerical_pipeline
from config import PROJECT_CONFIG


def main() -> None:
    ensure_project_directories()

    print("Running numerical regression pipeline...")
    numerical_results = run_numerical_pipeline()

    print("Running image classification pipeline...")
    image_results = run_image_pipeline()

    combined_results = {
        "numerical": numerical_results,
        "image": image_results,
    }
    save_json(combined_results, PROJECT_CONFIG["outputs"]["reports_dir"] / "complete_results.json")

    print("Building project documents...")
    build_project_documents(combined_results)

    print("Project run completed successfully.")
    print("Results, figures, reports, trained models, and documents were saved in the outputs/ and docs/ folders.")


if __name__ == "__main__":
    main()
