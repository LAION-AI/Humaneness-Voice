"""Launch the pinned local LFM3 Omni GGUF server; no credentials or public listener."""
import os
from pathlib import Path

home=Path(os.environ.get('HF_HOME','/mnt/nvme/moss-15-v2/.cache/lfm3'))
model=home/'hub/models--LiquidAI--LFM3-Omni-2B-GGUF/snapshots/5bcc26f7968f71d26e066027a70978529dbb605d'
server=os.environ.get('LFM3_OMNI_SERVER','/mnt/nvme/moss-15-v2/lfm3-llama-cpp/build/bin/llama-server')
argv=[server,'-m',str(model/'LFM3-Omni-2B-Q4_K_M.gguf'),'--mmproj',str(model/'mmproj-LFM3-Omni-2B-Q8_0.gguf'),
      '--host','127.0.0.1','--port',os.environ.get('LFM3_OMNI_PORT','18811'),'--n-gpu-layers','99','--ctx-size','4096','--batch-size','512','--ubatch-size','256',
      '--parallel','1','--threads','8','--no-mmproj-offload','--flash-attn','off','--reasoning','off','--no-webui']
env=dict(os.environ);env.setdefault('CUDA_VISIBLE_DEVICES','0');os.execvpe(server,argv,env)
