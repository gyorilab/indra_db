"""
Format a set of INDRA Statements into an HTML-based format.
"""
import sys
import csv
from os.path import abspath, dirname, join
import json
from collections import defaultdict
from jinja2 import Template
from indra.sources import indra_db_rest


# Create a template object from the template file, load once
template_path = join(dirname(abspath(__file__)), 'template.html')
with open(template_path, 'rt') as f:
    template_str = f.read()
    template = Template(template_str)


def format_evidence_text(edge, agent_list, evidence):
    """Highlight subject and object in raw text strings."""
    if ev.text is None:
        return '(None available)'

    raw_text = evidence.text
    ag_name_list = [ag.name for ag in agent_list]
    subj_ag_index = ag_name_list.index(edge[0])
    obj_ag_index = ag_name_list.index(edge[1])
    subj_text = evidence.annotations['agents']['raw_text'][subj_ag_index]
    obj_text = evidence.annotations['agents']['raw_text'][obj_ag_index]

    # TODO: highlight all instances of each gene, not just the first
    format_text = raw_text
    for ix, ag_text in enumerate((subj_text, obj_text)):
        if ag_text is None:
            continue
        if ix == 0:
            marker = '<span class="label label-danger">'
        else:
            marker = '<span class="label label-primary">'
        ag_start_ix = format_text.find(ag_text)
        ag_end_ix = ag_start_ix + len(ag_text)
        format_text = (format_text[0:ag_start_ix] + marker +
                       format_text[ag_start_ix:ag_end_ix] + '</span>' +
                       format_text[ag_end_ix:])
    return format_text


def format_statements(stmt_json):
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
    return f"""
<html><head></head><body>
<h1>Hello, API</h1>
<pre>{json.dumps(stmt_json, indent=4)}
</body></html>"""
