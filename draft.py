import json

# Đọc file ipynb
with open(r"C:\Users\Admin\Desktop\New folder\Hybrid_model.ipynb", "r", encoding="utf-8") as f:
    notebook = json.load(f)

with open("extracted_with_markdown.py", "w", encoding="utf-8") as f:
    for i, cell in enumerate(notebook["cells"], start=1):
        f.write(f"# ===== Cell {i} ({cell['cell_type']}) =====\n")
        
        if cell["cell_type"] == "code":
            code = "".join(cell["source"])
            f.write(code + "\n\n")
        elif cell["cell_type"] == "markdown":
            md = "".join(cell["source"])
            # Đưa markdown thành comment
            md_lines = md.splitlines()
            for line in md_lines:
                f.write("# " + line + "\n")
            f.write("\n")