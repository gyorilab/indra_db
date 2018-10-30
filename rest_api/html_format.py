"""
Format a set of INDRA Statements into an HTML-based format.
"""
from indra.statements import *
from indra.assemblers.html import HtmlAssembler

def format_statements(result):
    """Format the statements as HTML."""
    stmts_json = result.pop('statements')
    stmts_formatted = []
    stmts = stmts_from_json(stmts_json.values())
    html_assembler = HtmlAssembler(stmts, result)
    html_output = html_assembler.make_model()
    return html_output

