"""Server-only authenticated encryption; plaintext keys never leave this module.

Owner credentials persist outside the source tree. The separate master key is
protected by OS permissions; this is at-rest encryption, not a hardware vault.
Optional personal credentials remain encrypted in session memory only.
"""
import json,os,re,stat
from pathlib import Path

def validate_key(value):
    if not isinstance(value,str) or not re.fullmatch(r'sk-or-v1-[A-Za-z0-9_-]{32,200}',value):
        raise ValueError('Invalid OpenRouter key format')
    return value

class CredentialStore:
    def __init__(self,root=None):
        self.root=Path(root or os.environ.get('ROBO_CREDENTIAL_DIR',str(Path.home()/'.config/robby-credentials')))
    def _read(self,path):
        if path.is_symlink():raise RuntimeError('Credential symlinks are forbidden')
        info=path.stat()
        if info.st_uid!=os.getuid() or stat.S_IMODE(info.st_mode)&0o077:
            raise RuntimeError('Credential files require owner-only permissions')
        return path.read_bytes()
    def _write(self,path,raw):
        if self.root.is_symlink():raise RuntimeError('Credential directory cannot be a symlink')
        self.root.mkdir(mode=0o700,parents=True,exist_ok=True);os.chmod(self.root,0o700)
        temp=path.with_suffix('.new')
        fd=os.open(temp,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        try:
            with os.fdopen(fd,'wb') as f:f.write(raw)
            os.replace(temp,path)
        finally:
            if temp.exists():temp.unlink()
    def _fernet(self):
        from cryptography.fernet import Fernet
        path=self.root/'master.key'
        if not path.exists():self._write(path,Fernet.generate_key())
        return Fernet(self._read(path))
    def configured(self):return (self.root/'openrouter.enc').is_file()
    def seal(self,value):return self._fernet().encrypt(validate_key(value).encode()).decode()
    def open(self,token):return validate_key(self._fernet().decrypt(token.encode()).decode())
    def store_owner(self,value):self._write(self.root/'openrouter.enc',self.seal(value).encode())
    def owner(self):
        if not self.configured():raise RuntimeError('OpenRouter credential is not configured')
        return self.open(self._read(self.root/'openrouter.enc').decode())

def store():return CredentialStore()

if __name__=='__main__':
    import getpass
    value=getpass.getpass('OpenRouter key (hidden): ')
    store().store_owner(value)
    print('OpenRouter credential encrypted and saved outside the repository.')
