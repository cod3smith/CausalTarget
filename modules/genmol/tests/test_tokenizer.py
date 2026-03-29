"""Tests for the SMILES tokenizer."""

import pytest
from modules.genmol.data.tokenizer import (
    SmilesTokenizer,
    PAD_TOKEN,
    SOS_TOKEN,
    EOS_TOKEN,
    UNK_TOKEN,
)


# ── Sample SMILES ───────────────────────────────────────────────────
SAMPLE_SMILES = [
    "CCO",                    # ethanol
    "c1ccccc1",               # benzene
    "CC(=O)Oc1ccccc1C(O)=O",  # aspirin
    "CC(=O)NC1=CC=C(O)C=C1",  # acetaminophen
    "ClC1=CC=CC=C1",          # chlorobenzene
    "BrC1=CC=CC=C1",          # bromobenzene
]


@pytest.fixture
def tokenizer():
    tok = SmilesTokenizer(max_length=50)
    tok.build_vocab(SAMPLE_SMILES)
    return tok


class TestTokenization:
    """Test character-level SMILES tokenization."""

    def test_single_char_tokens(self):
        tokens = SmilesTokenizer.tokenize("CCO")
        assert tokens == ["C", "C", "O"]

    def test_multi_char_tokens(self):
        tokens = SmilesTokenizer.tokenize("ClC")
        assert tokens == ["Cl", "C"]

    def test_bromine(self):
        tokens = SmilesTokenizer.tokenize("BrC")
        assert tokens == ["Br", "C"]

    def test_ring_closure(self):
        tokens = SmilesTokenizer.tokenize("c1ccccc1")
        assert tokens == ["c", "1", "c", "c", "c", "c", "c", "1"]


class TestVocabulary:
    """Test vocabulary building."""

    def test_vocab_includes_special_tokens(self, tokenizer):
        assert PAD_TOKEN in tokenizer.token_to_idx
        assert SOS_TOKEN in tokenizer.token_to_idx
        assert EOS_TOKEN in tokenizer.token_to_idx
        assert UNK_TOKEN in tokenizer.token_to_idx

    def test_special_token_indices(self, tokenizer):
        assert tokenizer.pad_idx == 0
        assert tokenizer.sos_idx == 1
        assert tokenizer.eos_idx == 2
        assert tokenizer.unk_idx == 3

    def test_vocab_size(self, tokenizer):
        assert tokenizer.vocab_size > 4  # at least specials + some chars
        assert tokenizer.vocab_size == len(tokenizer.token_to_idx)

    def test_vocab_includes_cl_br(self, tokenizer):
        assert "Cl" in tokenizer.token_to_idx
        assert "Br" in tokenizer.token_to_idx


class TestEncodeDecode:
    """Test encode/decode round-trip."""

    def test_encode_length(self, tokenizer):
        ids = tokenizer.encode("CCO", pad=True)
        assert len(ids) == tokenizer.max_length

    def test_encode_starts_with_sos(self, tokenizer):
        ids = tokenizer.encode("CCO")
        assert ids[0] == tokenizer.sos_idx

    def test_encode_has_eos(self, tokenizer):
        ids = tokenizer.encode("CCO")
        # EOS should be after the tokens, before padding
        non_pad = [i for i in ids if i != tokenizer.pad_idx]
        assert non_pad[-1] == tokenizer.eos_idx

    def test_round_trip(self, tokenizer):
        for smi in SAMPLE_SMILES:
            ids = tokenizer.encode(smi)
            decoded = tokenizer.decode(ids)
            assert decoded == smi, f"Round-trip failed: {smi} → {decoded}"

    def test_encode_no_pad(self, tokenizer):
        ids = tokenizer.encode("CCO", pad=False)
        assert len(ids) == 5  # <SOS> C C O <EOS>

    def test_unknown_token(self, tokenizer):
        # Z is unlikely to be in a typical SMILES vocab
        ids = tokenizer.encode("Z")
        # Should use UNK token
        assert tokenizer.unk_idx in ids


class TestPersistence:
    """Test save/load round-trip."""

    def test_save_load(self, tokenizer, tmp_path):
        path = tmp_path / "tokenizer.json"
        tokenizer.save(path)

        loaded = SmilesTokenizer.load(path)
        assert loaded.vocab_size == tokenizer.vocab_size
        assert loaded.max_length == tokenizer.max_length

        # Check encode/decode still works
        for smi in SAMPLE_SMILES:
            ids_orig = tokenizer.encode(smi)
            ids_loaded = loaded.encode(smi)
            assert ids_orig == ids_loaded

    def test_repr(self, tokenizer):
        r = repr(tokenizer)
        assert "SmilesTokenizer" in r
        assert str(tokenizer.vocab_size) in r
