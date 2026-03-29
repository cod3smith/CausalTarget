"""
SMILES Tokenizer
==================

A character-level tokenizer for SMILES (Simplified Molecular-Input
Line-Entry System) strings.

Why character-level?
--------------------
SMILES strings encode molecular graphs as text.  Each character
typically represents an atom (``C``, ``N``, ``O``), a bond (``=``,
``#``), or structural information (``(``, ``)``, ring-closure digits).

Character-level tokenization is the simplest approach and works well
for VAE-based generative models because:

* The vocabulary is tiny (~45 characters) — easy to learn.
* The model can discover chemical syntax through reconstruction.
* No domain-specific segmentation rules are needed.

Multi-character tokens like ``Br``, ``Cl``, ``@@``, ``%10`` are
handled by a regex-based splitter that groups them correctly so
bromine isn't tokenized as ``B`` + ``r``.

Special tokens
--------------
* ``<PAD>`` — padding for batch collation (index 0)
* ``<SOS>`` — start-of-sequence, prepended to decoder input
* ``<EOS>`` — end-of-sequence, appended to signal generation should stop
* ``<UNK>`` — unknown character fallback
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Regex for SMILES token splitting ────────────────────────────────
# Matches multi-character tokens first (Cl, Br, @@, %NN), then single chars.
# Order matters: longer patterns must come first so "Cl" is matched
# before "C" + "l".
_SMILES_TOKENIZE = re.compile(
    r"Br|Cl|@@|@|\%\d{2}|."
)

# ── Special tokens ──────────────────────────────────────────────────
PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"
SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]


class SmilesTokenizer:
    """Character-level SMILES tokenizer with special tokens.

    The tokenizer builds a vocabulary from training data and provides
    ``encode`` / ``decode`` methods to convert between SMILES strings
    and integer sequences suitable for neural network input.

    Parameters
    ----------
    max_length : int
        Maximum sequence length (including ``<SOS>`` and ``<EOS>``).
        Longer SMILES are truncated.  Default 120 covers >99% of
        drug-like molecules in ChEMBL.

    Examples
    --------
    >>> tok = SmilesTokenizer()
    >>> tok.build_vocab(["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(O)=O"])
    >>> ids = tok.encode("CCO")
    >>> tok.decode(ids)
    'CCO'
    """

    def __init__(self, max_length: int = 120):
        self.max_length = max_length
        self.token_to_idx: dict[str, int] = {}
        self.idx_to_token: dict[int, str] = {}
        self.vocab_size: int = 0
        self._built = False

    # ── Vocabulary construction ─────────────────────────────────────

    def build_vocab(self, smiles_list: list[str]) -> None:
        """Build vocabulary from a list of SMILES strings.

        Scans all SMILES to discover the unique character set, then
        creates a mapping ``token → index``.  Special tokens occupy
        indices 0–3.

        Parameters
        ----------
        smiles_list : list[str]
            Training SMILES strings.
        """
        tokens: set[str] = set()
        for smi in smiles_list:
            tokens.update(self.tokenize(smi))

        # Sort for deterministic ordering
        sorted_tokens = sorted(tokens)

        self.token_to_idx = {}
        self.idx_to_token = {}

        # Special tokens first
        for i, special in enumerate(SPECIAL_TOKENS):
            self.token_to_idx[special] = i
            self.idx_to_token[i] = special

        # Regular tokens
        offset = len(SPECIAL_TOKENS)
        for i, tok in enumerate(sorted_tokens):
            if tok not in self.token_to_idx:
                idx = i + offset
                self.token_to_idx[tok] = idx
                self.idx_to_token[idx] = tok

        # Compact: re-index to remove gaps
        self.token_to_idx = {}
        self.idx_to_token = {}
        all_tokens = list(SPECIAL_TOKENS) + [
            t for t in sorted_tokens if t not in SPECIAL_TOKENS
        ]
        for i, tok in enumerate(all_tokens):
            self.token_to_idx[tok] = i
            self.idx_to_token[i] = tok

        self.vocab_size = len(self.token_to_idx)
        self._built = True

        logger.info(
            "Vocabulary built: %d tokens (including %d special).",
            self.vocab_size,
            len(SPECIAL_TOKENS),
        )

    # ── Tokenization ────────────────────────────────────────────────

    @staticmethod
    def tokenize(smiles: str) -> list[str]:
        """Split a SMILES string into individual tokens.

        Handles multi-character tokens (``Br``, ``Cl``, ``@@``, ``%NN``)
        correctly.

        Parameters
        ----------
        smiles : str
            A SMILES string.

        Returns
        -------
        list[str]
            List of character-level tokens.
        """
        return _SMILES_TOKENIZE.findall(smiles)

    # ── Encode / Decode ─────────────────────────────────────────────

    def encode(
        self,
        smiles: str,
        add_sos: bool = True,
        add_eos: bool = True,
        pad: bool = True,
    ) -> list[int]:
        """Encode a SMILES string into a list of integer token IDs.

        Parameters
        ----------
        smiles : str
            Input SMILES.
        add_sos : bool
            Prepend ``<SOS>`` token.
        add_eos : bool
            Append ``<EOS>`` token.
        pad : bool
            Right-pad with ``<PAD>`` to ``max_length``.

        Returns
        -------
        list[int]
            Integer token IDs.
        """
        assert self._built, "Call build_vocab() before encoding."

        tokens = self.tokenize(smiles)
        unk_idx = self.token_to_idx[UNK_TOKEN]

        ids: list[int] = []
        if add_sos:
            ids.append(self.token_to_idx[SOS_TOKEN])

        for tok in tokens:
            ids.append(self.token_to_idx.get(tok, unk_idx))

        if add_eos:
            ids.append(self.token_to_idx[EOS_TOKEN])

        # Truncate if too long
        if len(ids) > self.max_length:
            ids = ids[: self.max_length]
            # Ensure EOS at end
            if add_eos:
                ids[-1] = self.token_to_idx[EOS_TOKEN]

        # Pad to max_length
        if pad:
            pad_idx = self.token_to_idx[PAD_TOKEN]
            while len(ids) < self.max_length:
                ids.append(pad_idx)

        return ids

    def decode(
        self,
        ids: list[int],
        strip_special: bool = True,
    ) -> str:
        """Decode a list of integer token IDs back to a SMILES string.

        Parameters
        ----------
        ids : list[int]
            Token IDs.
        strip_special : bool
            Remove special tokens (``<PAD>``, ``<SOS>``, ``<EOS>``,
            ``<UNK>``) from the output.

        Returns
        -------
        str
            Decoded SMILES string.
        """
        assert self._built, "Call build_vocab() before decoding."

        special_indices = {
            self.token_to_idx[t] for t in SPECIAL_TOKENS
        }

        tokens: list[str] = []
        for idx in ids:
            if idx in self.idx_to_token:
                tok = self.idx_to_token[idx]
                if strip_special and idx in special_indices:
                    if tok == EOS_TOKEN:
                        break  # stop at EOS
                    continue
                tokens.append(tok)

        return "".join(tokens)

    # ── Persistence ─────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save vocabulary to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "max_length": self.max_length,
            "token_to_idx": self.token_to_idx,
        }
        path.write_text(json.dumps(data, indent=2))
        logger.info("Tokenizer saved → %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "SmilesTokenizer":
        """Load a tokenizer from a saved JSON file."""
        path = Path(path)
        data = json.loads(path.read_text())

        tok = cls(max_length=data["max_length"])
        tok.token_to_idx = data["token_to_idx"]
        tok.idx_to_token = {int(v): k for k, v in tok.token_to_idx.items()}
        tok.vocab_size = len(tok.token_to_idx)
        tok._built = True

        logger.info("Tokenizer loaded ← %s (%d tokens).", path, tok.vocab_size)
        return tok

    # ── Properties ──────────────────────────────────────────────────

    @property
    def pad_idx(self) -> int:
        """Index of the ``<PAD>`` token."""
        return self.token_to_idx[PAD_TOKEN]

    @property
    def sos_idx(self) -> int:
        """Index of the ``<SOS>`` token."""
        return self.token_to_idx[SOS_TOKEN]

    @property
    def eos_idx(self) -> int:
        """Index of the ``<EOS>`` token."""
        return self.token_to_idx[EOS_TOKEN]

    @property
    def unk_idx(self) -> int:
        """Index of the ``<UNK>`` token."""
        return self.token_to_idx[UNK_TOKEN]

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return (
            f"SmilesTokenizer(vocab_size={self.vocab_size}, "
            f"max_length={self.max_length})"
        )
