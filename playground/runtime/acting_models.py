"""Pinned full-FT Acting releases; no implicit checkpoint/adapter substitution."""
from pathlib import Path

import settings

MODELS = {
    'large17': {
        'id':'large17', 'label':'Large 1.7 · merged BF16', 'kind':'large',
        'repo':'laion/Humaneness-Voice-Large-1.7',
        'revision':'1c708b7ea106e097c44c48a58a194991c6749b4a',
        'checkpoint':'model.safetensors.index.json',
        'sha256':'4caf21d4613731bd40fab873a115f43c5962aac6e951c2e69b13a1285b058e61', 'native_root':True,
        'cache_dir':__import__('os').environ.get('HVS_LARGE17_CACHE',str(Path(settings.HF_CACHE)/'hub')),
        'shard_sha256':{
            'model-00001-of-00003.safetensors':'3a58abc85d557836f30046ba8be2e677dde15004f84bf8ee849edd80fab56292',
            'model-00002-of-00003.safetensors':'2e84194e1b60781129cc9efa0b792a08535e0a5027df9ae5ce4a1cb24cb04393',
            'model-00003-of-00003.safetensors':'0c7873c4426f18f1367f659c5815c284abd4e3673f8270c358f8dddc556d5df6'},
        'batch_size':1, 'max_batch_size':8, 'search_max_n':8,
        'reference_min_seconds':5.04, 'reference_max_seconds':14.96,
    },
    'small_rl': {
        'id':'small_rl','label':'Small 1.1 Alpha RL · r128 v1 · Default','kind':'small',
        'repo':'laion/Humaneness-Voice-Small-1.1-Alpha-RL-r128-v1',
        'revision':'182bc700cac29b2d69295a33398b13e1ca7fba69',
        'checkpoint':'checkpoints/step-00006109/model.safetensors',
        'sha256':'6fe4a757c505732d376a1116a3e5dc130fa5af845e9a9dc8474c2cb4f73b67c6',
        'adapter_sha256':'b441554a730000f96617958d73e74474ad83ffde486763750b4a1fd5b52da379',
        'rl_bundle':True,'policy_dtype':'fp32','step':6109,'rl_updates':2434,
        'cache_dir':str(settings.RL_CACHE),'batch_size':1,'max_batch_size':8,'search_max_n':8,
    },
    'large_rl': {
        'id':'large_rl','label':'Large 1.1 Alpha RL · r128 v1','kind':'large',
        'repo':'laion/Humaneness-Voice-Large-1.1-Alpha-RL-r128-v1',
        'revision':'7f050dc0e5c1b2415258c357c1edc27e1452cf07',
        'checkpoint':'checkpoints/step-00003055/model.safetensors',
        'sha256':'c7d42ba93fdd0ee6acb0ef7b97da224b681510819f925ae85c4edf3848b29f8e',
        'adapter_sha256':'113f4fb4598efcc909dd7ea11422842d5c0dc61e20519ed4d9af5d028ea668d4',
        'rl_bundle':True,'policy_dtype':'fp32','step':3055,'rl_updates':2370,
        'cache_dir':str(settings.RL_CACHE),'batch_size':1,'max_batch_size':2,'search_max_n':2,
    },
    'large_acting': {
        'id': 'large_acting', 'label': 'Large Acting · Step 432', 'kind': 'large',
        'repo': 'laion/Humaneness-Voice-Large-Acting',
        'revision': 'd64c8d833495ed4a18d5e7abbe468d9e2fc4fa4e',
        'checkpoint': 'checkpoints/step-00000432/model.safetensors',
        'sha256': '9a6e7400bb53dc7dc4cdcae2495f2c31ec70b564a99cb8dbb394fab76991931d',
        'step': 432, 'batch_size': 1, 'max_batch_size': 4, 'search_max_n': 4,
    },
    'small_acting': {
        # Stable UI/API identifier; its former Acting weights are replaced.
        'id': 'small_acting', 'label': 'Small · 12 Oct 2026', 'kind': 'small',
        'repo': 'laion/Humaneness-Voice-Small',
        'revision': '3365015e53329ae3ac3a4bb3d541ed760418797c',
        'checkpoint': 'checkpoints/HVS-Small-2026-10-12/model_bf16.pt',
        'sha256': '45f1de4bcc62bef038afc812ac8681b3e6507d9d97e4c18d27518f5a5a346bd4',
        'format': 'pytorch_state_dict', 'stage': 'HVS-Small-2026-10-12',
        'step': 15238, 'merged_training_adapter_step': 2298,
        'batch_size': 1, 'max_batch_size': 8, 'search_max_n': 8,
    },
    'small_alpha': {
        'id':'small_alpha','label':'Small 1.1 Alpha · Step 3,775 · Experimental','kind':'small',
        'repo':'laion/Humaneness-Voice-Small-1.1-Alpha',
        'revision':'aed9e955dae4591922b159561bd995fd9b288ece',
        'checkpoint':'intermediate-checkpoints/step-00003775/model.safetensors',
        'sha256':'ab7229198123135e0aa5964f155794fa6b6ef9bdfadd98c4356705f35b05b5a5',
        'stage':'1.1 Alpha interim','step':3775,'total_steps':6109,
        'experimental':True,'training_complete':False,
        'batch_size':1,'max_batch_size':8,'search_max_n':8,
    },
    'large_alpha': {
        'id':'large_alpha','label':'Large 1.1 Alpha · Step 2,377 · Experimental','kind':'large',
        'repo':'laion/Humaneness-Voice-Large-1.1-Alpha',
        'revision':'ababe387f10f663057dad9ddb4652422200acab3',
        'checkpoint':'intermediate-checkpoints/step-00002377/model.safetensors',
        'sha256':'d63bfeb2951e05cd3a82fb1ddf80ff2634fdc7519ac1520374e7bdfec17c2cb5',
        'stage':'1.1 Alpha interim','step':2377,'total_steps':3055,
        'cache_dir':str(settings.EXPERIMENTAL_LARGE_CACHE),
        'experimental':True,'training_complete':False,
        'batch_size':1,'max_batch_size':4,'search_max_n':4,
    },
}


def catalog():
    from checkpoint_ram import RAM
    return {'default': settings.DEFAULT_MODEL, 'models': list(MODELS.values()),
            'policy': 'Verified private BF16 CPU weight cache; Large 1.7 uses its already-merged release once. RL precision follows the configured mode. Only the selected TTS model is in VRAM.',
            'ram_cache':RAM.describe(),
            'steering_available': False, 'adapters_available': False}


def download(model_id):
    from huggingface_hub import snapshot_download
    from rank1_adapters import sha256
    release = MODELS[model_id]
    if release.get('native_root'):
        root=Path(snapshot_download(release['repo'],revision=release['revision'],
            cache_dir=release['cache_dir'],token=False,allow_patterns=[
                '*.py','*.json','*.txt','*.md','LICENSE','NOTICE*','UPSTREAM*','model-*.safetensors']))
        for name,expected in {release['checkpoint']:release['sha256'],**release['shard_sha256']}.items():
            if sha256(root/name)!=expected:
                raise ValueError('Large 1.7 integrity check failed: '+name)
        return root
    if release.get('rl_bundle'):
        bundle=Path(snapshot_download(release['repo'],revision=release['revision'],
            cache_dir=release['cache_dir'],token=False,allow_patterns=[
                'base/code/*.py','base/assets/**','base/'+release['checkpoint'],
                'adapter_config.json','adapter.safetensors','README.md','BUNDLE.md','code/adapters.py']))
        root=bundle/'base'
        if sha256(root/release['checkpoint'])!=release['sha256']:
            raise ValueError('RL bundled base integrity check failed: '+model_id)
        if sha256(bundle/'adapter.safetensors')!=release['adapter_sha256']:
            raise ValueError('RL adapter integrity check failed: '+model_id)
        import json
        config=json.loads((bundle/'adapter_config.json').read_text())
        if config.get('format')!='humaneness-moss-attention-lora-v1' or config.get('rank')!=128:
            raise ValueError('Unsupported RL adapter configuration')
        return root
    root = Path(snapshot_download(release['repo'], revision=release['revision'],
        cache_dir=release.get('cache_dir',str(Path(settings.HF_CACHE) / 'hub')), token=False,
        allow_patterns=['code/*.py', 'assets/**', release['checkpoint'],
                        str(Path(release['checkpoint']).parent / 'READY.json'),
                        str(Path(release['checkpoint']).parent / 'metadata.json'),
                        'README.md', 'LICENSE', 'model_kind.txt', 'infer.py']))
    if sha256(root / release['checkpoint']) != release['sha256']:
        raise ValueError(f"Acting checkpoint integrity check failed: {model_id}")
    if release.get('format')=='pytorch_state_dict':
        import json
        metadata=json.loads((root/Path(release['checkpoint']).parent/'metadata.json').read_text())
        if metadata.get('model_sha256')!=release['sha256'] or metadata.get('checkpoint')!=release['stage']:
            raise ValueError('Small release metadata does not match pinned checkpoint')
    if release.get('experimental'):
        import json
        ready=json.loads((root/Path(release['checkpoint']).parent/'READY.json').read_text())
        if ready.get('step')!=release['step']:
            raise ValueError('Experimental READY proof does not match pinned checkpoint')
    return root


if __name__ == '__main__':
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as pool:
        for root in pool.map(download, MODELS):
            print('Verified Acting snapshot:', root, flush=True)
