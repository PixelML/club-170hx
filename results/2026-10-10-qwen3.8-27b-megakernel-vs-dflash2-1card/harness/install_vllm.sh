set -euxo pipefail
printf '#!/bin/sh\nexec "$@"\n' > /usr/local/bin/sudo; chmod +x /usr/local/bin/sudo
cd /work; [ -d club-170hx ] || git clone -q https://github.com/PixelML/club-170hx.git
git -C club-170hx rev-parse HEAD > /work/club-170hx.sha
export RECIPE_DIR=/work/club-170hx/recipes/qwen3.8-27b-dflash2 RUNTIME_ROOT=/work/qwen-serving
python3 - <<'PY' > /work/recipe_install.sh
import json; nb=json.load(open('/work/club-170hx/recipes/qwen3.8-27b-dflash2/reproduce.ipynb'))
for c in nb['cells']:
    s=''.join(c['source'])
    if "install = r'''" in s:
        print(s.split("install = r'''",1)[1].split("'''",1)[0])
PY
bash /work/recipe_install.sh
$RUNTIME_ROOT/venv/bin/python -c "import vllm,torch;print('VLLM',vllm.__version__,torch.__version__)"
echo VLLM_INSTALL_DONE
