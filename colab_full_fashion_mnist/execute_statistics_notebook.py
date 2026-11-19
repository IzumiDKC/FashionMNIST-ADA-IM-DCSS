"""Execute the deliverable on local results and export a readable HTML copy."""
import json
import sys
import zipfile
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from nbconvert import HTMLExporter

project = Path(__file__).resolve().parent.parent
path = project / "ThongKe_DCSS_FashionMNIST.ipynb"
data = project / "DCSS_FashionMNIST_Full"
before = {str(p.relative_to(data)): (p.stat().st_size, p.stat().st_mtime_ns) for p in data.rglob("*") if p.is_file()}
notebook = nbformat.read(path, as_version=4)
manager = KernelManager(kernel_name="python3")
manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
client = NotebookClient(notebook, timeout=1200, km=manager, resources={"metadata": {"path": str(project)}})
try:
    client.execute()
finally:
    nbformat.write(notebook, path)
    if manager.has_kernel:
        manager.shutdown_kernel(now=True)
after = {str(p.relative_to(data)): (p.stat().st_size, p.stat().st_mtime_ns) for p in data.rglob("*") if p.is_file()}
assert before == after, "Input experiment files were modified"
errors = [o for c in notebook.cells if c.cell_type == "code" for o in c.outputs if o.output_type == "error"]
assert not errors
exporter = HTMLExporter()
exporter.exclude_input = True
exporter.exclude_input_prompt = True
exporter.exclude_output_prompt = True
html, _ = exporter.from_notebook_node(notebook)
(project / "DCSS_Statistics" / "ThongKe_DCSS_FashionMNIST.html").write_text(html, encoding="utf-8")
proof = {"executed_notebook": path.name, "code_cells_executed": sum(c.cell_type == "code" for c in notebook.cells),
         "error_outputs": len(errors), "input_files_unchanged": len(before), "training_executed": False,
         "images_in_notebook": sum("image/png" in o.get("data", {}) for c in notebook.cells if c.cell_type == "code" for o in c.outputs)}
(project / "DCSS_Statistics" / "notebook_validation.json").write_text(json.dumps(proof, indent=2), encoding="utf-8")
from analyze_results import export_bundle
archive = export_bundle({"output": project / "DCSS_Statistics"})
with zipfile.ZipFile(archive, "a", zipfile.ZIP_DEFLATED) as bundle:
    bundle.write(path, path.name)
print(json.dumps(proof, indent=2))
