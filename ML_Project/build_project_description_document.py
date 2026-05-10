from src.common.project_description_builder import build_project_description_document


if __name__ == "__main__":
    outputs = build_project_description_document()
    print("Project Description Document generated successfully.")
    for label, path in outputs.items():
        print(f"{label}: {path}")
