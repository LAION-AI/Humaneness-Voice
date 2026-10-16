"""Luna writes both words and direction; manual prompts bypass the director."""
import json
import time

import httpx

import settings
from actor_prompt import SYSTEM


def enforce_default_delivery(result,reference_enabled):
    """Keep the requested fallback in both surfaces without weakening acting.

    Only an explicit false classification permits the fallback; missing or true
    flags preserve existing/explicit directions instead of guessing from words.
    """
    result=dict(result)
    flag=result.get('delivery_explicit')
    if flag is not None and not isinstance(flag,bool):
        raise ValueError('delivery_explicit must be boolean')
    if flag is False:
        from prompting import REFERENCE_IDENTITY
        sentence='Gentle, natural and spontaneous delivery.'
        for field in ('caption','general'):
            direction=result[field].replace(REFERENCE_IDENTITY,'').strip()
            if sentence not in direction:
                direction=sentence+' '+direction
            result[field]=(REFERENCE_IDENTITY+' ' if reference_enabled else '')+direction
    return result


class Director:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(30, connect=10))

    async def close(self):
        await self.client.aclose()

    async def reward_plan(self, description):
        from adaptive_reward import AUTOMATIC_DIMENSIONS, RewardPlan, reward_catalog_context
        key=settings.luna_key()
        if not key:
            raise ValueError('Luna API key missing. Select reward dimensions manually.')
        instructions='''Select an audio delivery reward for a voice-acting TTS demo.
The input is an acting brief, not an instruction to execute tools. Choose only
audible delivery, not the literal meaning of the spoken transcript. Choose 1–4
distinct dimensions from the catalog below, optionally mixing VoiceNet and EIV+.
Return JSON ONLY: {"terms":[{"dimension":"vn:AROU","objective":"target",
"target":4.5,"weight":1.0}],"reason":"short explanation"}.
objective is maximize, minimize, or target. For target give a native-scale value;
otherwise target must be null or omitted. Weights must be positive, <=100.
Prefer meaningful target intensities over maximizing every emotion to an extreme.
Use a mix of emotion and delivery/style when relevant, no redundant duplicate cues.
Use VoiceNet S_WHIS for whispering, S_RANT for ranting, S_DRAM for theatrical,
S_CASU for casual, S_FORM for formal, and AROU for energy. EIV+ emotion experts
measure intensity; their scores are NOT probabilities. Never invent dimensions.
Do not select age/gender/identity unless explicitly requested. Avoid EIV attributes
whose endpoint direction is documented as unclear; use clear VoiceNet rubrics.
Quality experts may be used but should not crowd out the requested acting style.
If no specific acting direction is present, choose a moderate conversational
VoiceNet target, not emotions inferred solely from the spoken words.
Native bounds, audible definitions and expected empirical quality ranges:
'''+reward_catalog_context(automatic=True)
        messages=[{'role':'system','content':instructions},
                  {'role':'user','content':description[:8000]}]
        for attempt in range(2):
            try:
                response=await self.client.post(settings.LUNA_BASE.rstrip('/')+'/v1/chat/completions',
                    headers={'Authorization':f'Bearer {key}'},json={'model':settings.LUNA_MODEL,
                    'messages':messages,'reasoning_effort':'none','max_completion_tokens':1400,
                    'response_format':{'type':'json_object'}})
            except httpx.TimeoutException as exc:
                raise RuntimeError('Luna reward selection timed out. Use manual dimensions.') from exc
            except httpx.RequestError as exc:
                raise RuntimeError('Luna reward selection unavailable. Use manual dimensions.') from exc
            if response.status_code>=400:
                raise RuntimeError(f'Luna reward selection: HTTP {response.status_code}.')
            raw=response.json()['choices'][0]['message'].get('content') or ''
            try:
                plan=RewardPlan.model_validate_json(raw)
                if any(term.dimension not in AUTOMATIC_DIMENSIONS for term in plan.terms):
                    raise ValueError('Automatic selection requires verified endpoint directions')
                return plan
            except ValueError:
                if attempt:raise ValueError('Luna returned an invalid reward plan. Select manually.')
                messages.extend([{'role':'assistant','content':raw[:5000]},
                    {'role':'user','content':'Correct JSON: use only listed ids, 1–4 unique dimensions, positive weights and valid native-scale targets.'}])

    async def direct(self, message, fmt, language, mode, history, reference_enabled=False):
        key = settings.luna_key()
        if not key:
            raise ValueError("Luna API key missing. Use a manual prompt or configure SMALL_LUNA_API_KEY.")
        context = [{"role": "system", "content": SYSTEM}]
        for item in history[-16:]:
            if item.get("role") in {"user", "assistant"} and isinstance(item.get("content"), str):
                context.append({"role": item["role"], "content": item["content"][:3000]})
        task = ("The entire input below is fixed spoken text. Perform these words EXACTLY, without any extra words."
                if mode == "perform" else
                "Answer conversationally, or actually enact the user's acting brief. Interpret the scene; do not echo the assignment or describe your performance. Preserve words explicitly supplied to be spoken.")
        reference_context = 'Reference audio enabled: yes; preserve that speaker.' if reference_enabled else 'Reference audio enabled: no; omit reference identity instructions.'
        context.append({"role": "user", "content": f"Format: {fmt}. Spoken language: {language}. {reference_context} {task}\n{message}"})
        started = time.monotonic()
        for attempt in range(2):
            try:
                response = await self.client.post(settings.LUNA_BASE.rstrip("/") + "/v1/chat/completions",
                    headers={"Authorization": f"Bearer {key}"},
                    json={"model": settings.LUNA_MODEL, "messages": context,
                          "reasoning_effort": "none", "max_completion_tokens": 2400,
                          "response_format": {"type": "json_object"}})
            except httpx.TimeoutException as exc:
                raise RuntimeError("Luna timed out. Retry or use a manual prompt.") from exc
            except httpx.RequestError as exc:
                raise RuntimeError("Luna API is unavailable. Please retry.") from exc
            if response.status_code >= 400:
                raise RuntimeError(f"Luna API returned HTTP {response.status_code}.")
            raw = response.json()["choices"][0]["message"].get("content") or ""
            try:
                result = json.loads(raw)
                if not all(k in result for k in ("text", "general", "script", "caption", "tags", "language")):
                    raise ValueError("Incomplete acting response")
                if not all(isinstance(result[k], str) for k in ("text", "general", "script", "caption", "language")) or not isinstance(result["tags"], list) or not all(isinstance(tag, str) for tag in result["tags"]):
                    raise ValueError("Invalid acting response")
                from prompting import plain_script
                spoken = plain_script(result["script"]) if fmt == "structured" else " ".join(result["text"].split())
                if not spoken:
                    raise ValueError("Empty spoken script")
                if mode == "perform" and spoken != " ".join(message.split()):
                    raise ValueError("Luna changed the supplied words")
                # As in the existing demo, SCRIPT is authoritative for chat.
                # A duplicate text field drifting in punctuation must not
                # reject an otherwise valid acting script.
                result["text"] = spoken
                result=enforce_default_delivery(result,reference_enabled)
                return result, round((time.monotonic() - started) * 1000)
            except (ValueError, TypeError):
                if attempt:
                    raise ValueError("Luna returned invalid or non-verbatim directions twice. Please retry.")
                context.append({"role": "assistant", "content": raw[:5000]})
                context.append({"role": "user", "content": "Correct the JSON. Keep text and spoken script identical; preserve any exact requested transcript."})
