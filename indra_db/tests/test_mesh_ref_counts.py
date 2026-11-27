from types import SimpleNamespace

import pytest
from sqlalchemy import Column, Integer, BigInteger
from sqlalchemy.orm import declarative_base

from indra_db.client.readonly.mesh_ref_counts import get_mesh_ref_counts

Base = declarative_base()


class _FakeMeshConceptRefCounts(Base):
    __tablename__ = 'mesh_concept_ref_counts'
    mk_hash = Column(BigInteger, primary_key=True)
    mesh_num = Column(Integer, primary_key=True)
    ref_count = Column(Integer)
    pmid_count = Column(Integer)


class _FakeMeshTermRefCounts(Base):
    __tablename__ = 'mesh_term_ref_counts'
    mk_hash = Column(BigInteger, primary_key=True)
    mesh_num = Column(Integer, primary_key=True)
    ref_count = Column(Integer)
    pmid_count = Column(Integer)


class _FakeQuery:
    def __init__(self, rows):
        self._rows = rows

    def filter(self, *args, **kwargs):
        return self

    def group_by(self, *args, **kwargs):
        return self

    def having(self, *args, **kwargs):
        return self

    def all(self):
        return self._rows


class _FakeSession:
    def __init__(self, concept_rows, term_rows):
        self._concept_rows = concept_rows
        self._term_rows = term_rows

    def query(self, *entities):
        table = entities[0].class_
        if table is _FakeMeshConceptRefCounts:
            return _FakeQuery(self._concept_rows)
        if table is _FakeMeshTermRefCounts:
            return _FakeQuery(self._term_rows)
        raise AssertionError(f"Unexpected table: {table}")


def _make_fake_ro(concept_rows, term_rows):
    return SimpleNamespace(
        MeshConceptRefCounts=_FakeMeshConceptRefCounts,
        MeshTermRefCounts=_FakeMeshTermRefCounts,
        session=_FakeSession(concept_rows, term_rows),
    )


def test_mixed_concept_and_term_total_preserves_pmid_count():
    # Statement 100 has 10 distinct PMIDs total (pmid_count=10 in both tables),
    # with 3 PMIDs for C0001243 and 4 + 5 PMIDs for D000225 and D002352.
    ro = _make_fake_ro(
        concept_rows=[(100, [1243], [3], 10)],
        term_rows=[(100, [225, 2352], [4, 5], 10)],
    )

    res = get_mesh_ref_counts(['C0001243', 'D000225', 'D002352'], ro=ro)

    assert res == {
        100: {
            'C0001243': 3,
            'D000225': 4,
            'D002352': 5,
            'total': 10,
        }
    }


def test_mixed_concept_and_term_require_all():
    ro = _make_fake_ro(
        concept_rows=[
            (100, [1243], [3], 10),
            (200, [1243], [2], 7),
        ],
        term_rows=[
            (100, [225], [4], 10),
            (300, [225], [1], 5),
        ],
    )

    res = get_mesh_ref_counts(
        ['C0001243', 'D000225'], require_all=True, ro=ro
    )

    assert res == {
        100: {
            'C0001243': 3,
            'D000225': 4,
            'total': 10,
        }
    }


def test_invalid_mesh_prefix_raises():
    ro = _make_fake_ro([], [])
    with pytest.raises(ValueError, match="C or D"):
        get_mesh_ref_counts(['X000123'], ro=ro)
