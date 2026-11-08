"""Explicit conditioning/audit language routing for the five Podcast languages."""
LANGUAGES={'en':'English','de':'German','nl':'Dutch','fr':'French','es':'Spanish'}
LANGUAGE_SEEDS={'en':0,'de':10000,'nl':20000,'fr':30000,'es':40000}
QWEN_ALIGNMENT_LANGUAGES={'en','de','fr','es'}
def language_name(code):
    try:return LANGUAGES[code]
    except KeyError:raise ValueError(f'Unsupported Podcast language: {code}') from None
