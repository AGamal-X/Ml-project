from src.common.discussion_guide_builder import build_discussion_guide


if __name__ == "__main__":
    markdown_path, pdf_path = build_discussion_guide()
    print("Discussion guide generated successfully.")
    print(f"Markdown: {markdown_path}")
    print(f"PDF: {pdf_path}")
