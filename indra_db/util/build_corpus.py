"""Dump test corpora of content covering all REACH rules

This script is designed select content from the database based on the REACH
rules that have been triggered within that content. Three slightly different
methods are used, and three corpora are produces, each as a directory.
"""

import os
import json
from indra_db.util import unpack
from indra_db.util import get_ro, get_db

db = get_db('primary')

rs = db.select_all(db.RawStatements, db.Reading.reader == 'REACH', 
                   db.RawStatements.reading_id == db.Reading.id, yield_per=10000)
found_by = {}
for r in rs:
    found_by[r.id] = json.loads(r.json)['evidence'][0]['annotations']['found_by']
    
fb_set = set(found_by.values())
print(f"Found {len(fb_set)} distinct found-by rules.")

fb_counts = {}
for sid, word in found_by.items():
    fb_counts[word] = fb_counts.get(word, 0) + 1
    
fb_sids = {}
for sid, word in found_by.items():
    if word not in fb_sids:
        fb_sids[word] = []
    fb_sids[word].append(sid)
    
tc_data = db.select_all([db.TextContent.id, db.TextContent.source, db.TextContent.text_type, db.RawStatements.id],
                        db.Reading.reader == 'REACH', *db.link(db.TextContent, db.RawStatements))
tc_lookup = {sid: (tcid, src, tt) for tcid, src, tt, sid in tc_data}

fb_tc_dict = {}
tc_fb_dict = {}
for fb, sids in sorted(fb_sids.items(), key=lambda t: len(t[1])):
    print(fb, len(sids))
    this_dict = {}
    for sid in sids:
        tcid, src, tt = tc_lookup[sid]
        
        # Add fb to lookup by tcid
        if tcid not in tc_fb_dict:
            tc_fb_dict[tcid] = set()
        tc_fb_dict[tcid].add(fb)
        
        # Add tcid sid data to list of content with this fb.
        key = (src, tt)
        if key not in this_dict:
            this_dict[key] = []
        this_dict[key].append({'tcid': tcid, 'sid': sid})
    fb_tc_dict[fb] = this_dict
    
        
num_with = 0
for fb, cont_meta in fb_tc_dict.items():
    if ('pubmed', 'abstract') not in cont_meta and ('pubmed', 'title') not in cont_meta:
        print(f"{fb:70} {fb_counts[fb]} {cont_meta.keys()}")
    else:
        num_with += 1
        
ranking = [('pubmed', 'abstract'), ('pmc_oa', 'fulltext'), ('manuscripts', 'fulltext'), ('pubmed', 'title')]
        

def dump_tcs(tcids, dirname):
    tcs = db.select_all([db.TextRef.id, db.TextRef.pmid, db.TextRef.pmcid, db.TextContent.id, 
                         db.TextContent.source, db.TextContent.text_type, db.TextContent.content], 
                        db.TextContent.id.in_(tcids), *db.link(db.TextRef, db.TextContent))
    tt_counts = {}
    for row in tcs:
        tt = row[-1]
        tt_counts[tt] = tt_counts.get(tt, 0) + 1
        
    print(dirname, tt_counts)

    if not os.path.exists(dirname):
        os.mkdir(dirname)
    else:
        raise ValueError(f"Directory {dirname} already exists.")

    metadata = {}
    for trid, pmid, pmcid, tcid, src, tt, cont_bytes in tcs:
        metadata[tcid] = {'trid': trid, 'pmid': pmid, 'tcid': tcid, 'pmcid': pmcid, 'source': src, 'text_type': tt}
        if src == 'pubmed':
            fmt = 'txt'
        else:
            fmt = 'nxml'
        with open(f'{dirname}/{tcid}.{fmt}', 'w') as f:
            f.write(unpack(cont_bytes))
    with open(f'{dirname}/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)


# Select strictly the content with the most rules represented. No preference
# based on type.
corpus_ids = []
rep_fbs = set()
for fb, cont_meta in sorted(fb_tc_dict.items(), key=lambda t: fb_counts[t[0]]):
    print("--------------------------------------------")
    print("Examining rule:", fb, fb_counts[fb])
    if fb in rep_fbs:
        print("Already represented...skipping")
        continue
        
    best_ref = None
    for text_cat, text_list in cont_meta.items():
        print(text_cat, len(text_list))
        counted_refs = [(len(tc_fb_dict[d['tcid']] - rep_fbs), d['tcid']) for d in text_list]
        print(f"best ref for {text_cat}:", max(counted_refs))
        if best_ref is None:
            best_ref = max(counted_refs)
        else:
            this_ref = max(counted_refs)
            if this_ref > best_ref:
                best_ref = this_ref
    print(f"Overall best ref for {fb}:", best_ref)
    corpus_ids.append(best_ref[1])
    rep_fbs |= tc_fb_dict[best_ref[1]]
    print(len(rep_fbs))
    if len(rep_fbs) == len(fb_counts):
        print("DONE!")
        break
dump_tcs(corpus_ids, 'corpus_1')

        
# Select the content with most rules, with the preference for abstract as a tie-breaker.
corpus_ids_2 = []
rep_fbs = set()
for fb, cont_meta in sorted(fb_tc_dict.items(), key=lambda t: fb_counts[t[0]]):
    print("--------------------------------------------")
    print("Examining rule:", fb, fb_counts[fb])
    if fb in rep_fbs:
        print("Already represented...skipping")
        continue
        
    all_counted_refs = []
    for text_cat, text_list in cont_meta.items():
        print(text_cat, len(text_list))
        all_counted_refs += [(len(tc_fb_dict[d['tcid']] - rep_fbs), -ranking.index(text_cat), d['tcid']) for d in text_list]
    best_ref = max(all_counted_refs)
    print(f"Overall best ref for {fb}:", best_ref)
    corpus_ids_2.append(best_ref[-1])
    rep_fbs |= tc_fb_dict[best_ref[-1]]
    print(len(rep_fbs))
    if len(rep_fbs) == len(fb_counts):
        print("DONE!")
        break
dump_tcs(corpus_ids_2, 'corpus_2')
        

# Select abstracts whenever possible, fulltext only when necessary.
corpus_ids_3 = []
rep_fbs = set()
for fb, cont_meta in sorted(fb_tc_dict.items(), key=lambda t: fb_counts[t[0]]):
    print("--------------------------------------------")
    print("Examining rule:", fb, fb_counts[fb])
    if fb in rep_fbs:
        print("Already represented...skipping")
        continue
        
    all_counted_refs = []
    for text_cat, text_list in cont_meta.items():
        print(text_cat, len(text_list))
        all_counted_refs += [(-ranking.index(text_cat), len(tc_fb_dict[d['tcid']] - rep_fbs), d['tcid']) for d in text_list]
    best_ref = max(all_counted_refs)
    print(f"Overall best ref for {fb}:", best_ref)
    corpus_ids_3.append(best_ref[-1])
    rep_fbs |= tc_fb_dict[best_ref[-1]]
    print(len(rep_fbs))
    if len(rep_fbs) == len(fb_counts):
        print("DONE!")
        break
dump_tcs(corpus_ids_3, 'corpus_3')    
    
