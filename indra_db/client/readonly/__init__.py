from .util import *
from .query import *


def get_source_info():
    from indra.config import KNOWLEDGE_SOURCE_INFO
    from indra_db import get_ro
    ro = get_ro('primary')

    ro_srcs: set = ro.get_source_names()
    sources = {}
    for src_id in ro_srcs:
        src_info = {'id': src_id}
        if src_id == 'vhn':
            indra_src_info = KNOWLEDGE_SOURCE_INFO['virhostnet']
        elif src_id == 'bel_lc':
            indra_src_info = KNOWLEDGE_SOURCE_INFO['bel']
        elif src_id == 'pe':
            indra_src_info = KNOWLEDGE_SOURCE_INFO['phosphoelm']
        elif src_id == 'psp':
            indra_src_info = {
                'name': 'Phosphosite Plus',
                'link': 'https://www.phosphosite.org/homeAction.action',
                'type': 'database',
                'domain': 'biology'
            }
        else:
            indra_src_info = KNOWLEDGE_SOURCE_INFO[src_id]

        src_info.update(indra_src_info)

        if src_id == 'eidos':
            src_info['domain'] = 'biology'

        sources[src_id] = src_info
    return sources
