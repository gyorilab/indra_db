from .util import *
from .query import *


def get_ro_source_info():
    from indra.sources import SOURCE_INFO
    from indra_db import get_ro
    ro = get_ro('primary')

    ro_srcs: set = ro.get_source_names()
    sources = {}
    for src_id in ro_srcs:
        src_info = {'id': src_id}
        lookup_id = src_id
        if src_id == 'vhn':
            lookup_id = 'virhostnet'
        elif src_id == 'bel_lc':
            lookup_id = 'bel'
        elif src_id == 'pe':
            lookup_id = 'phosphoelm'
        elif src_id == 'psp':
            lookup_id = 'phosphosite'

        src_info.update(SOURCE_INFO[lookup_id])

        if src_id == 'eidos':
            src_info['domain'] = 'biology'

        sources[src_id] = src_info
    return sources
