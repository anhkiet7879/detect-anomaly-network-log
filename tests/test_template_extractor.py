"""
test_template_extractor.py — kiểm tra [H4]: Reset State và khóa từ điển Drain3.
Chạy (trong thư mục N3.10):   python -m pytest tests\\test_template_extractor.py -v
"""
import os
import sys

import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from templates.log_template_extractor import EMPTY_ID, NEW_ID, LogTemplateExtractor  # noqa: E402

CONFIG = os.path.join(ROOT, "config", "drain3.ini")
TRAIN = ["generating core.12", "generating core.345", "data TLB error interrupt",
         "ciod: failed to read message prefix on control stream (CioStream socket to 172.16.96.116:35728",
         "", "generating core.7"]
TEST = ["generating core.999", "a totally unseen message here", ""]


@pytest.fixture
def ext():
    e = LogTemplateExtractor(CONFIG)
    e.fit(TRAIN)
    return e


def test_same_template_after_masking(ext):
    ids, new = ext.transform(["generating core.1", "generating core.2"])
    assert ids[0] == ids[1] and new.sum() == 0


def test_dictionary_unchanged_after_transform(ext):
    before = ext.fingerprint()
    ext.transform(TEST * 100)
    assert ext.fingerprint() == before


def test_unseen_and_empty(ext):
    ids, new = ext.transform(TEST)
    assert ids[1] == NEW_ID and new[1] == 1
    assert ids[2] == EMPTY_ID and new[2] == 0


def test_learn_blocked_after_fit(ext):
    with pytest.raises(RuntimeError):
        ext.learn("generating core.1")


def test_reset_gives_fresh_state(ext):
    n = ext.n_templates
    ext.fit(["only one message"])           # fit() tự reset
    assert ext.n_templates == 1
    ext.fit(TRAIN)
    assert ext.n_templates == n
