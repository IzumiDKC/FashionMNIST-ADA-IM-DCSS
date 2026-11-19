"""Execute report mode locally with explicit substitutes for Colab auth/mount only."""
from pathlib import Path
import ast
import json
import nbformat
from nbclient import NotebookClient
from IPython.core.inputtransformer2 import TransformerManager
import dcss_core as core

ROOT=Path(__file__).parent.resolve()


def main():
    source=ROOT/"DCSS_ADA_IM_MultiDataset.ipynb"
    notebook=nbformat.read(source,as_version=4)
    nbformat.validate(notebook)
    transform=TransformerManager()
    code_cells=[c for c in notebook.cells if c.cell_type=="code"]
    for c in code_cells:
        ast.parse(transform.transform_cell(c.source))
    local=nbformat.reads(nbformat.writes(notebook),as_version=4)
    local_dir=ROOT/"audit/notebook_execution/modules"
    local_dir.mkdir(parents=True,exist_ok=True)
    old=ROOT.parent/"DCSS_FashionMNIST_Full"
    old_hashes={str(p.relative_to(old)):core.digest(p) for p in old.rglob("*") if p.is_file()}
    output_parent=ROOT/"audit/notebook_execution"
    for cell in local.cells:
        if cell.cell_type!="code":
            continue
        if cell.source.startswith("%%writefile /content/"):
            header,body=cell.source.split("\n",1)
            name=header.rsplit("/",1)[-1]
            cell.source=f"from pathlib import Path\nPath({str(local_dir/name)!r}).write_bytes({body!r}.encode('utf-8'))"
            continue
        if "from google.colab import drive, auth" in cell.source:
            start=cell.source.index('from pathlib import Path')
            cell.source=cell.source[start:].replace('RUN_TRAINING = True','RUN_TRAINING = False')
        elif "WORKSPACE = data.drive_mounted_path" in cell.source:
            cell.source=cell.source.replace('WORKSPACE = data.drive_mounted_path(drive_service,WORKSPACE_FOLDER_ID)',f'WORKSPACE = Path({str(output_parent)!r})')
            cell.source=cell.source.replace('FASHION_ROOT = data.drive_mounted_path(drive_service,FASHION_RESULTS_FOLDER_ID)',f'FASHION_ROOT = Path({str(old)!r})')
        cell.source=cell.source.replace('/content',local_dir.as_posix())
    try:
        NotebookClient(local,timeout=240,kernel_name="python3",resources={"metadata":{"path":str(ROOT)}},allow_errors=False).execute()
    except Exception:
        nbformat.write(local,ROOT/"audit/notebook_failed_diagnostic.ipynb")
        errors=[(i,o.get("ename"),o.get("evalue")) for i,c in enumerate(local.cells) for o in c.get("outputs",[]) if o.output_type=="error"]
        print("Notebook execution errors:",errors,flush=True)
        raise RuntimeError("Local notebook validation failed; inspect saved diagnostic (no full source printed)") from None
    nbformat.write(local,ROOT/"audit/notebook_report_mode_executed.ipynb")
    after={str(p.relative_to(old)):core.digest(p) for p in old.rglob("*") if p.is_file()}
    assert after==old_hashes,"Historical Fashion results must be byte-identical"
    result=dict(notebook_sha256=core.digest(source),schema_valid=True,code_cells_compiled=len(code_cells),
                local_report_mode_executed=True,fashion_files_unchanged=len(after),
                substitutions=["Colab auth/mount replaced with verified local paths","/content replaced with local audit directory","writefile writes identical UTF-8/LF bytes explicitly on Windows","RUN_TRAINING=False"],
                gpu_executed=False,colab_oauth_executed=False,new_full_experiments_executed=False)
    core.atomic_json(ROOT/"audit/notebook_validation.json",result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    main()
