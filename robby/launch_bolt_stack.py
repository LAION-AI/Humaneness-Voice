"""Reproducible launcher for llama.cpp, optional Robby ASR, and FastAPI.

The shared Humaneness TTS server must already run at ROBO_TTS_BASE. This launcher
only terminates children it created. It never stops an existing GPU service.
"""
import argparse,json,os,signal,socket,subprocess,sys,time
from pathlib import Path
from huggingface_hub import hf_hub_download
from bolt_director import MODEL_REPO,MODEL_REVISION

ROOT=Path(__file__).resolve().parent
FILE='Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf'

def port_used(port):
    with socket.socket() as s:return s.connect_ex(('127.0.0.1',port))==0

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--llama-server',type=Path,required=True)
    p.add_argument('--models',type=Path,default=Path('/mnt/nvme/robby-bolt/models'))
    p.add_argument('--download',action='store_true');p.add_argument('--gpu',default='0')
    p.add_argument('--with-robo',action='store_true',help='Start the ASR/avatar server too; port 8793 must be free')
    p.add_argument('--reuse-llm',action='store_true',help='Use an already running compatible local server at port 8799')
    p.add_argument('--with-asr-models',action='store_true',help='Start the optional CPU-preloaded ASR bank on port 8801')
    args=p.parse_args();model=args.models/FILE
    if not args.llama_server.is_file():p.error('llama-server binary does not exist')
    if not model.exists() and args.download:
        hf_hub_download(MODEL_REPO,FILE,revision=MODEL_REVISION,local_dir=args.models)
    if not model.is_file():p.error('Model not found; pass --download or supply --models')
    if port_used(8798):p.error('FastAPI port 8798 is in use; leave the existing service running or stop its owned process first')
    if args.with_robo and port_used(8793):p.error('ASR port 8793 is in use; this launcher will not replace its owner')
    if args.with_asr_models and port_used(8801):p.error('ASR model bank port 8801 is already in use')
    if not args.reuse_llm and port_used(8799):p.error('LLM port 8799 is in use; use --reuse-llm to retain that server')
    env=dict(os.environ,ROBO_DIRECTOR_MODE='bolt',ROBO_SENTENCE_STREAM='1',ROBO_BOLT_CHUNK_FRAMES='20',
        ROBO_BOLT_BASE='http://127.0.0.1:8799',ROBO_DIRECTOR_BASE='http://127.0.0.1:8799',
        ROBO_DIRECTOR_MODEL='nemotron-bolt',ROBO_DIRECTOR_API_KEY='local')
    logs=ROOT/'logs';logs.mkdir(exist_ok=True);children=[];files=[]
    def spawn(label,command,extra=None):
        log=(logs/(label+'.log')).open('ab');files.append(log)
        child=subprocess.Popen(command,cwd=ROOT,env={**env,**(extra or {})},stdout=log,stderr=subprocess.STDOUT)
        children.append((label,child));print(f'{label}: PID {child.pid}',flush=True)
    try:
        if not args.reuse_llm:
            spawn('bolt-llama',[str(args.llama_server),'-m',str(model),'--alias','nemotron-bolt',
                '--host','127.0.0.1','--port','8799','-ngl','999','-c','16384','-np','2',
                '-fa','on','-b','512','-ub','256','-t','6','--jinja','--no-webui'],{'CUDA_VISIBLE_DEVICES':args.gpu})
        if args.with_robo:spawn('bolt-robo',[sys.executable,'-u','server.py'],{'CUDA_VISIBLE_DEVICES':args.gpu})
        if args.with_asr_models:spawn('asr-model-worker',[sys.executable,'-u','asr_model_worker.py'],{'CUDA_VISIBLE_DEVICES':args.gpu,'ROBO_ASR_DEVICE':'cuda:0'})
        spawn('bolt-api',[sys.executable,'-m','uvicorn','bolt_api:app','--host','127.0.0.1','--port','8798'])
        print('UI: http://127.0.0.1:8798/desk-robot/ · API: http://127.0.0.1:8798/docs',flush=True)
        while True:
            for label,child in children:
                if child.poll() is not None:raise RuntimeError(f'{label} exited with status {child.returncode}; inspect logs/{label}.log')
            time.sleep(.5)
    except KeyboardInterrupt:pass
    finally:
        for _,child in reversed(children):
            if child.poll() is None:child.terminate()
        for _,child in children:
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:print(f'PID {child.pid} is still finishing its active session.',flush=True)
        for log in files:log.close()

if __name__=='__main__':main()
