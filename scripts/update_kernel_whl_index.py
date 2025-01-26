# Reference: https://github.com/flashinfer-ai/flashinfer/blob/v0.2.0/scripts/update_whl_index.py

import hashlib
import pathlib
import re

for path in sorted(pathlib.Path("sgl-kernel/dist").glob("*.whl")):
    with open(path, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()
    ver = re.findall(r"sgl_kernel-([0-9.]+(?:\.post[0-9]+)?)-", path.name)[0]
    index_dir = pathlib.Path(f"sgl-whl/cu118/sgl-kernel")
    index_dir.mkdir(exist_ok=True)
    base_url = "https://github.com/sgl-project/whl/releases/download"
    full_url = f"{base_url}/v{ver}/{path.name}#sha256={sha256}"
    with (index_dir / "index.html").open("a") as f:
        f.write(f'<a href="{full_url}">{path.name}</a><br>\n')