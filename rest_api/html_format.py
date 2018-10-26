"""
Format a set of INDRA Statements into an HTML-based format.
"""
import sys
import csv
from os.path import abspath, dirname, join
import re
import json
from collections import defaultdict
from jinja2 import Template
from indra.sources import indra_db_rest
from indra.assemblers.english import EnglishAssembler
from indra.statements import stmts_from_json
from indra.databases import get_identifiers_url

# Create a template object from the template file, load once
template_path = join(dirname(abspath(__file__)), 'template.html')
with open(template_path, 'rt') as f:
    template_str = f.read()
    template = Template(template_str)

# TODO:
# - Highlight text in english assembled sentences
# - Highlight text in evidence sentences
# - For both, add links to identifiers.org



def format_evidence_text(stmt):
    """Highlight subject and object in raw text strings."""
    badge_list = ['primary', 'danger', 'success', 'info', 'warning']
    ev_list = []
    for ix, ev in enumerate(stmt.evidence):
        if ev.text is None:
            format_text = '(None available)'
        else:
            indices = []
            for ix, ag in enumerate(stmt.agent_list()):
                ag_text = ev.annotations['agents']['raw_text'][ix]
                if ag_text is None:
                    continue
                # Build up a set of indices
                indices += [(m.start(), m.start() + len(ag_text), ag_text, ix)
                            for m in re.finditer(re.escape(ag_text), ev.text)]
            # Sort the indices by their start position
            indices.sort(key=lambda x: x[0])
            # Now, add the marker text for each occurrence of the strings
            format_text = ''
            start_pos = 0
            tag_start = '<span class="label label-primary">'
            tag_close = '</span>'
            for i, j, ag_text, ag_ix in indices:
                # Get the tag with the correct badge
                tag_start = '<span class="label label-%s">' % badge_list[ag_ix]
                tag_close = '</span>'
                # Add the text before this agent, if any
                format_text += ev.text[start_pos:i]
                # Add wrapper for this entity
                format_text += tag_start + ag_text + tag_close
                # Now set the next start position
                start_pos = j
            # Add the last section of text
            format_text += ev.text[start_pos:]
        ev_list.append({'source_api': ev.source_api,
                        'pmid': ev.pmid, 'text': format_text })
    return ev_list


def format_statements(result):
    #stmt_json = result.pop('statements')
    #stmt_str = json.dumps(stmt_json) if stmt_json else "No statements."

    """
    for subj, obj in edges:
        for stmt in stmts:
            for ev in stmt.evidence:
                #content[stmt][(subj, obj)].append(stmt.evidence)
                content[(subj, obj)].append({
                  'source_api': ev.source_api,
                  'pmid': ev.pmid,
                  'text': format_evidence_text((subj, obj), stmt.agent_list(), ev)
               })
    """
    stmts_formatted = []
    # Each entry in the statement list will have
    # - English assembled top level
    # - (optional INDRA Statement raw)
    # - HTML formatted evidence (with clickable entity names?)

    stmts_json = result.pop('statements')
    for stmt_hash, stmt_json in stmts_json.items():
        stmt = stmts_from_json([stmt_json])[0] # TODO: Need to go back to stmt?
        ea = EnglishAssembler([stmt])
        english = ea.make_model()
        ev_list = format_evidence_text(stmt)
        total_evidence = result['evidence_totals'][int(stmt_hash)]
        stmts_formatted.append({
            'hash': stmt_hash,
            'english': english,
            'evidence': ev_list,
            'evidence_returned': len(ev_list),
            'total_evidence': total_evidence})
    return template.render(statements=stmts_formatted, **result)
    #return f"<pre>{json.dumps(result['statements'], indent=4)}</pre>"
