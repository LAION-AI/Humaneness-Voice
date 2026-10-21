"""Fetch only the pinned optional Laya source, without overwriting an existing checkout."""
import subprocess
from pathlib import Path
from laya_policy import SOURCE_COMMIT

def main():
    root=Path(__file__).resolve().parent;target=root/'vendor/laya'
    if target.exists():
        sha=subprocess.check_output(['git','-C',str(target),'rev-parse','HEAD'],text=True).strip()
        if sha!=SOURCE_COMMIT:raise SystemExit('Existing Laya checkout differs; preserved unchanged. Choose a separate approved checkout.')
        print('Pinned Laya source already present.');return
    target.parent.mkdir(exist_ok=True)
    subprocess.run(['git','clone','--filter=blob:none','--no-checkout','https://github.com/NandhaKishorM/laya.git',str(target)],check=True)
    subprocess.run(['git','-C',str(target),'fetch','--depth','1','origin',SOURCE_COMMIT],check=True)
    subprocess.run(['git','-C',str(target),'checkout','--detach',SOURCE_COMMIT],check=True)
    print('Pinned source ready. manage_laya.py start downloads only the multilingual checkpoint, not the whole family.')

if __name__=='__main__':main()
