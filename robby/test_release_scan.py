import re,unittest
from scan_staged_release import PATTERNS

class ReleaseScanTests(unittest.TestCase):
    def test_known_credential_families_and_dynamic_headers(self):
        synthetic={
            'hf_token':b'hf_'+b'A'*30,
            'github_token':b'ghp_'+b'B'*36,
            'provider_secret':b'sk-'+b'C'*30,
            'private_key':b'-----BEGIN '+b'RSA PRIVATE KEY-----',
            'literal_bearer':b'Bearer '+b'D'*32,
            'literal_credential':b'api_key="'+b'E'*32+b'"',
        }
        for name,value in synthetic.items():self.assertIsNotNone(re.search(PATTERNS[name],value),name)
        safe=b"headers={'Authorization':'Bearer '+key}; api_key='test'"
        self.assertFalse(any(re.search(p,safe) for p in PATTERNS.values()))
