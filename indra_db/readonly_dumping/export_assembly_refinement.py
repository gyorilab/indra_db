import os
import re

from indra import logger
from indra.ontology.bio import bio_ontology
from indra.preassembler import Preassembler
from indra_db.readonly_dumping.export_assembly import split_tsv_gz_file, \
    batch_size, count_rows_in_tsv_gz, get_refinement_graph, \
    refinement_cycles_fpath, calculate_belief
from indra_db.readonly_dumping.locations import refinements_fpath, \
    belief_scores_pkl_fpath, split_unique_statements_folder_fpath, \
    unique_stmts_fpath

if __name__ == '__main__':
    if not refinements_fpath.exists() or not belief_scores_pkl_fpath.exists():
        logger.info("6. Running setup for refinement calculation")
        # Setup bio ontology for pre-assembler
        bio_ontology.initialize()
        bio_ontology._build_transitive_closure()
        pa = Preassembler(bio_ontology)

        # 6. Calculate refinement graph:

        if not split_unique_statements_folder_fpath.exists():
            logger.info("Splitting unique statements")
            # time: 30 min
            split_tsv_gz_file(unique_stmts_fpath.as_posix(),
                              split_unique_statements_folder_fpath.as_posix(),
                              batch_size=batch_size)
            logger.info(
                "Finished splitting unique statement"
            )
        else:
            logger.info(
                "split_unique_statements_folder exist"
            )
        split_unique_files = [os.path.join(split_unique_statements_folder_fpath, f)
                              for f in
                              os.listdir(split_unique_statements_folder_fpath)
                              if f.endswith(".gz")]
        split_unique_files = sorted(
            split_unique_files,
            key=lambda x: int(re.findall(r'\d+', x)[0])
        )
        batch_count = len(split_unique_files)
        # get the n_rows in the last incompleted batch
        last_count = count_rows_in_tsv_gz(split_unique_files[-1])
        num_rows = (batch_count - 1) * batch_size + last_count
        logger.info(f"{num_rows} rows in unique statements with "
                    f"{batch_count} batches")
        cycles_found = False

        ref_graph = get_refinement_graph(n_rows=num_rows,
                                         split_files=split_unique_files)
        if cycles_found:
            logger.info(
                f"Refinement graph stored in variable 'ref_graph', "
                f"edges saved to {refinements_fpath.as_posix()}"
                f"and cycles saved to {refinement_cycles_fpath.as_posix()}"
            )

        else:
            # 7. Get belief scores, if there were no refinement cycles
            logger.info("7. Calculating belief")
            calculate_belief(
                ref_graph, num_batches=batch_count, batch_size=batch_size
            )
    else:
        logger.info("Final output already exists, stopping script")