from sqlalchemy import func

from indra_db import get_ro


def get_mesh_ref_counts(mesh_terms, require_all=False, ro=None):
    """Get the number of distinct pmids by mesh term for each hash.

    This function directly queries a table in the readonly database that counts
    the number of distinct PMIDs for each mesh term/hash pair. Given a list of
    mesh terms, this will return a dictionary keyed by hash containing
    dictionaries indicating how much support the hash has from each of the given
    mesh IDs in terms of distinct PMIDs (thus distinct publications).

    Parameters
    ----------
    mesh_terms : list
        A list of mesh term strings of the form "D000#####".
    require_all : Optional[bool]
        If True, require that each entry in the result includes both mesh terms.
        In other words, only return results where, for each hash, articles exist
        with support from all MeSH IDs given, not just one or the other. Default
        is False
    ro : Optional[DatabaseManager]
        A database manager handle. The default is the primary readonly, as
        indicated by environment variables or the config file.
    """
    # Get the default readonly database, if needed..
    if ro is None:
        ro = get_ro('primary')

    # Make sure the mesh IDs are of the correct kind.
    if not all(m.startswith('D') or m.startswith('C') for m in mesh_terms):
        raise ValueError("All mesh terms must begin with C or D.")

    # Convert the IDs to numbers for faster lookup.
    result = {}
    for prefix, table in [('C', ro.MeshConceptRefCounts),
                          ('D', ro.MeshTermRefCounts)]:
        mesh_num_map = {int(m[1:]): m for m in mesh_terms
                        if m.startswith(prefix)}
        if not mesh_num_map:
            continue

        # Build the query.
        nums = func.array_agg(table.mesh_num)
        counts = func.array_agg(table.ref_count)
        q = ro.session.query(table.mk_hash, nums.label('nums'),
                             counts.label('ref_counts'), table.pmid_count)
        if len(mesh_num_map.keys()) == 1:
            q = q.filter(table.mesh_num == list(mesh_num_map.keys())[0])
        elif len(mesh_num_map.keys()) > 1:
            q = q.filter(table.mesh_num.in_(mesh_num_map.keys()))
        q = q.group_by(table.mk_hash, table.pmid_count)

        # Apply the require all option by comparing the length of the nums array
        # to the number of inputs.
        if require_all:
            q = q.having(func.cardinality(nums) == len(mesh_num_map.keys()))

        # Parse the results.
        for mk_hash, nums, counts, pmid_count in q.all():
            count_dict = {mesh_num_map[mesh_num]: ref_count
                          for mesh_num, ref_count in zip(nums, counts)}
            if mk_hash not in result:
                result[mk_hash] = count_dict
                result[mk_hash]['total'] = pmid_count
            else:
                result[mk_hash].update(count_dict)
                result[mk_hash]['total'] += sum(counts)

    # Little sloppy, but delete any that don't meet the require_all constraint.
    if require_all:
        num_terms = len(set(mesh_terms))
        for mk_hash in result.copy().keys():
            if len(result[mk_hash]) != num_terms + 1:
                result.pop(mk_hash)
    return result
