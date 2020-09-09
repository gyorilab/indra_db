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
    if not all(m.startswith('D') for m in mesh_terms):
        raise ValueError("All mesh terms must begin with D.")

    # Convert the IDs to numbers for faster lookup.
    mesh_num_map = {int(m[1:]): m for m in mesh_terms}

    # Build the query.
    t = ro.MeshRefCounts
    nums = func.array_agg(t.mesh_num)
    counts = func.array_agg(t.ref_count)
    q = ro.session.query(t.mk_hash, nums.label('nums'),
                         counts.label('ref_counts'), t.pmid_count)
    if len(mesh_num_map.keys()) == 1:
        q = q.filter(t.mesh_num == list(mesh_num_map.keys())[0])
    elif len(mesh_num_map.keys()) > 1:
        q = q.filter(t.mesh_num.in_(mesh_num_map.keys()))
    else:
        raise ValueError("Must submit at least one mesh term.")
    q = q.group_by(t.mk_hash, t.pmid_count)

    # Apply the require all option by comparing the length of the nums array
    # to the number of inputs.
    if require_all:
        q = q.having(func.cardinality(nums) == len(mesh_num_map.keys()))

    # Parse the results.
    result = {}
    for mk_hash, nums, counts, pmid_count in q.all():
        result[mk_hash] = {mesh_num_map[mesh_num]: ref_count
                           for mesh_num, ref_count in zip(nums, counts)}
        result[mk_hash]['total'] = pmid_count
    return result
