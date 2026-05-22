from indra_db.cli.content_aws import PMC
from indra_db.databases import texttypes

import pytest
from itertools import islice

from indra_db.cli.content_aws import _NihAwsClient


@pytest.mark.slow
def test_nih_aws_client_inventory():
    client = _NihAwsClient()

    manifest_key = client.get_latest_inventory_manifest_key()
    assert manifest_key.endswith("/manifest.json")

    csv_keys = client.get_inventory_csv_keys(manifest_key)
    assert len(csv_keys) > 0

    rows = list(islice(client.iter_article_versions_from_inventory(), 5))
    assert len(rows) == 5

    for pmcid, version, article_prefix, metadata_key in rows:
        assert pmcid.startswith("PMC")
        assert isinstance(version, int)
        assert article_prefix == f"{pmcid}.{version}"
        assert metadata_key == f"metadata/{article_prefix}.json"

        xml_key = client.get_xml_key(article_prefix)
        assert xml_key == f"{article_prefix}/{article_prefix}.xml"


def test_nih_aws_client_pmcid():
    client = _NihAwsClient()

    article_prefix = "PMC5443623.1"
    metadata_key = client.get_metadata_key(article_prefix)
    xml_key = client.get_xml_key(article_prefix)

    metadata = client.get_json(metadata_key)
    xml_str = client.get_file(xml_key)

    assert metadata["pmcid"] == "PMC5443623"
    assert metadata["version"] == 1
    assert "<article" in xml_str[:1000]


@pytest.mark.slow
def test_pmc_parser_single_pmcid():
    pmc = PMC()

    article_prefix = "PMC5443623.1"
    metadata_key = pmc.aws.get_metadata_key(article_prefix)
    xml_key = pmc.aws.get_xml_key(article_prefix)

    metadata = pmc.aws.get_json(metadata_key)
    xml_str = pmc.aws.get_file(xml_key)

    res = pmc.get_data_from_xml_str(
        xml_str,
        xml_key,
        metadata=metadata,
    )

    assert res is not None

    tr, tc = res

    assert tr["pmcid"] == "PMC5443623"
    assert tr["pmid"] == "26010632"
    assert tr["doi"] == "10.1001/JAMA.2015.5024"
    assert tr["manuscript_id"] == "NIHMS857042"
    assert tr["pub_year"] == 2015

    assert tc["pmcid"] == "PMC5443623"
    assert tc["text_type"] == texttypes.FULLTEXT
    assert tc["license"] is not None
    assert tc["content"] is not None

@pytest.mark.slow
def test_pmc_source():
    pmc = PMC()

    metadata = pmc.aws.get_json("metadata/PMC5443623.1.json")
    source = pmc.get_source_from_metadata(metadata)

    assert source in {"pmc_oa", "manuscripts"}
    assert source == "manuscripts"