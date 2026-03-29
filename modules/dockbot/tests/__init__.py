"""
DockBot Tests — Ligand Preparation
=====================================

Tests that the ligand preparation pipeline produces valid 3D molecules
and PDBQT output.  These tests use only RDKit (no Vina or network
access required).
"""

from __future__ import annotations

import pytest
from rdkit import Chem

from modules.dockbot.ligand_prep import (
    prepare_ligand,
    mol_to_pdbqt_string,
    prepare_ligand_pdbqt,
)


# ── Fixtures ────────────────────────────────────────────────────────

ASPIRIN = "CC(=O)Oc1ccccc1C(O)=O"
IBUPROFEN = "CC(C)Cc1ccc(cc1)[C@@H](C)C(O)=O"
CAFFEINE = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
INVALID = "not_a_smiles_string"


# ── Tests ───────────────────────────────────────────────────────────

class TestPrepareLigand:
    """Test 3D conformer generation."""

    def test_aspirin_produces_mol(self):
        mol = prepare_ligand(ASPIRIN, name="aspirin")
        assert mol is not None
        assert mol.GetNumConformers() >= 1

    def test_has_3d_coords(self):
        mol = prepare_ligand(ASPIRIN)
        assert mol is not None
        conf = mol.GetConformer()
        pos = conf.GetAtomPosition(0)
        # At least one coordinate should be non-zero for a 3D structure
        assert not (pos.x == 0.0 and pos.y == 0.0 and pos.z == 0.0)

    def test_has_hydrogens(self):
        mol = prepare_ligand(ASPIRIN)
        assert mol is not None
        # Aspirin has explicit H atoms after AddHs
        h_count = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 1)
        assert h_count > 0

    def test_name_stored(self):
        mol = prepare_ligand(ASPIRIN, name="aspirin")
        assert mol is not None
        assert mol.GetProp("_Name") == "aspirin"

    def test_smiles_stored(self):
        mol = prepare_ligand(ASPIRIN)
        assert mol is not None
        assert mol.HasProp("SMILES")

    def test_invalid_smiles_returns_none(self):
        mol = prepare_ligand(INVALID)
        assert mol is None

    def test_caffeine(self):
        mol = prepare_ligand(CAFFEINE, name="caffeine")
        assert mol is not None
        assert mol.GetNumConformers() >= 1

    def test_multiple_conformers(self):
        mol = prepare_ligand(ASPIRIN, num_conformers=5)
        assert mol is not None
        # After minimisation the best is kept
        assert mol.GetNumConformers() >= 1

    def test_no_minimize(self):
        mol = prepare_ligand(ASPIRIN, minimize=False)
        assert mol is not None
        assert mol.GetNumConformers() >= 1


class TestPDBQT:
    """Test PDBQT conversion."""

    def test_pdbqt_string_not_empty(self):
        mol = prepare_ligand(ASPIRIN)
        assert mol is not None
        pdbqt = mol_to_pdbqt_string(mol)
        assert len(pdbqt) > 0

    def test_pdbqt_contains_atom_lines(self):
        mol = prepare_ligand(ASPIRIN)
        assert mol is not None
        pdbqt = mol_to_pdbqt_string(mol)
        # Should contain ATOM or HETATM lines
        lines = pdbqt.splitlines()
        atom_lines = [l for l in lines if l.startswith(("ATOM", "HETATM"))]
        assert len(atom_lines) > 0


class TestPrepareLigandPDBQT:
    """Test the full convenience pipeline."""

    def test_returns_mol_and_pdbqt(self):
        mol, pdbqt = prepare_ligand_pdbqt(ASPIRIN, name="aspirin")
        assert mol is not None
        assert len(pdbqt) > 0

    def test_invalid_returns_none(self):
        mol, pdbqt = prepare_ligand_pdbqt(INVALID)
        assert mol is None
        assert pdbqt == ""

    def test_ibuprofen(self):
        mol, pdbqt = prepare_ligand_pdbqt(IBUPROFEN, name="ibuprofen")
        assert mol is not None
        assert "ATOM" in pdbqt or "HETATM" in pdbqt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
