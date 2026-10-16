"""Source layout checks + real frontend controller/server-schema agreement."""
from collections import Counter
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT=Path(__file__).resolve().parents[1]


class Elements(HTMLParser):
    def __init__(self,html):
        super().__init__();self.ids=[];self.attributes={};self.select=None;self.options={}
        self.feed(html)

    def handle_starttag(self,tag,attributes):
        attrs=dict(attributes)
        if 'id' in attrs:
            self.ids.append(attrs['id']);self.attributes[attrs['id']]=attrs
        if tag=='select':self.select=attrs.get('id');self.options[self.select]=[]
        if tag=='option' and self.select:self.options[self.select].append(attrs['value'])

    def handle_endtag(self,tag):
        if tag=='select':self.select=None


def test_ui_keeps_all_search_and_reward_controls_discoverable():
    html=(ROOT/'index.html').read_text();elements=Elements(html)
    assert all(count==1 for count in Counter(elements.ids).values())
    for id in ['generation-settings','input-settings','reference-settings','sampling-settings',
               'reward-settings','burst-settings','metrics-settings','configuration-summary',
               'searchn','searchchunk','searchseconds','rewardmode','burstmode','fileA','fileB']:
        assert id in elements.ids
    assert 'hidden' not in elements.attributes['searchoptions']
    assert 'open' in elements.attributes['generation-settings']
    assert elements.options['generation']==['single','best','all','search_winner','search_beam']
    assert elements.options['searchn']==[str(n) for n in range(1,17)]
    assert elements.options['searchchunk']==['10','20']
    assert elements.options['rewardmode']==['clap','adaptive_auto','adaptive_manual']
    assert elements.options['burstmode']==['off','location','typed']
    assert elements.attributes['temperature']['min']=='0'
    assert elements.attributes['g']['value']=='2'
    assert 'max-height:46vh' not in html
    assert '[hidden]{display:none!important}' in html
    for id in ['voice-choice','model-choice','checkpoint-summary']:
        assert id in elements.ids
    for id in ['lora-open','lora-open-sidebar','lora-dialog','lora-default','lora-search','lora-list']:
        assert id not in elements.ids
    assert elements.attributes['searchseconds']['value']=='30'
    assert 'Humaneness-Voice-Small-Rank1-LoRAs' not in html
    assert '<script src="/lora-ui.js"></script>' not in html
    assert 'data-lora-preset' not in html and 'body.adapters={}' in html


def test_actual_ui_controller_payloads_validate_against_server():
    node=os.environ.get('SMALL_UI_TEST_NODE') or shutil.which('node')
    if not node:pytest.skip('Node >=18 required for frontend controller checks')
    version=subprocess.check_output([node,'--version'],text=True).strip()
    if int(version.lstrip('v').split('.')[0])<18:
        pytest.skip('Set SMALL_UI_TEST_NODE to a Node >=18 binary')
    output=subprocess.check_output([node,str(ROOT/'ui_controller_test.cjs')],text=True,timeout=20)
    check=json.loads(output)
    from small_app import Turn
    assert check['search_combinations']==64
    for payload in check['payloads']:
        if payload['generation'].startswith('search_') and payload['search_n']>8:
            with pytest.raises(ValueError):Turn.model_validate(payload)
            continue
        turn=Turn.model_validate(payload)
        assert turn.generation==payload['generation']
    assert len(check['payloads'])>64


def test_playground_defaults_new_small_single_streaming_800ms():
    html=(ROOT/'index.html').read_text();elements=Elements(html)
    assert '<option value="single" selected>' in html
    assert '<option value="small_rl" selected>' in html
    assert elements.options['model-choice']==['small_rl','large_rl','small_acting','large_acting','small_alpha','large_alpha']
    assert 'Experimental · training in progress' in html
    assert 'model-ram-status' in elements.ids
    assert elements.attributes['n']['value']=='1'
    assert elements.attributes['model-batch']['value']=='1'
    assert '<option value="streaming" selected>' in html
    assert '<option value="10" selected>800 ms' in html
    assert "$('generation').value='single'" in html
    assert "$('n').value='1'" in html
    assert "$('streaming').value='streaming'" in html
    assert 'checked' not in elements.attributes['dynamic-reference']
    assert 'checked' not in elements.attributes['speaker-steering']


def test_actual_combined_lora_controller_payloads_validate_against_server():
    node=os.environ.get('SMALL_UI_TEST_NODE') or shutil.which('node')
    if not node:pytest.skip('Node >=18 required')
    version=subprocess.check_output([node,'--version'],text=True).strip()
    if int(version.lstrip('v').split('.')[0])<18:pytest.skip('Node >=18 required')
    output=subprocess.check_output([node,str(ROOT/'lora_controller_test.cjs')],text=True,timeout=20)
    result=json.loads(output)
    from dpo_adapters import AdapterSelection
    assert result['presets']==4 and result['old_adapter_preferences_not_migrated'] and result['default_no_adapter']
    for pick in result['payloads']:
        assert AdapterSelection.model_validate(pick).model_dump()==pick
