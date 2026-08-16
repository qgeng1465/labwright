"""Genomics-pipeline parameterization — ChAMP methylation & PLINK genotype batches.

A reviewer-facing Level-3 question is *pipeline parameterization*: given an
input cohort size, what batch parameters should a downstream bioinformatics
pipeline use? This module answers the two canonical cases:

**ChAMP (methylation array analysis in R)** — how many Illumina BeadChip arrays
and chips a cohort needs, and the array-content scale it will process:

* one array per sample; the BeadChip formats hold 12 (450K) or 8 (EPIC)
  samples per physical chip, so ``n_chips = ceil(n_samples / chip_size)``.
* array content (probe counts) is a published Illumina specification:
  ~485,577 probes for HumanMethylation450 and ~865,918 for MethylationEPIC.

**PLINK (genotype analysis)** — how the binary PLINK dataset scales with cohort
size and variant count, and how it splits per chromosome:

* the binary ``.bed`` stores 2 bits per sample per variant — 1 byte per 4
  samples — so ``.bed size = n_samples · n_variants / 4`` bytes.
* per-chromosome processing splits into the 25 standard files (1–22, X, Y,
  MT); given a per-chromosome variant count, the per-file ``.bed`` size follows
  from the same 2-bits rule.

These are *software conventions* (tool/format documentation), not literature
values — the docstrings label them as such. The "fail-rate" and "quality-pass"
quantities are parameters the caller supplies; nothing here claims a measured
QC yield.

References
----------
- Illumina HumanMethylation450 / MethylationEPIC BeadChip content & 12-/8-sample
  format: manufacturer product documentation (software/array convention).
- ChAMP: Morris, Butcher, Feber et al., Bioinformatics 30(2):267–268 (2014),
  doi:10.1093/bioinformatics/btt684 — vignette documents per-array processing.
- PLINK 1.9: Chang et al., GigaScience 4:7 (2015), doi:10.1186/s13742-015-0047-8 —
  binary ``.bed`` 2-bits/sample/variant format and per-chromosome files.
"""

from __future__ import annotations

import math

#: Illumina BeadChip sample capacity by platform (arrays per physical chip).
CHIP_CAPACITY: dict[str, int] = {"450k": 12, "epic": 8}

#: Published array content (probes) by platform — Illumina product spec.
ARRAY_PROBES: dict[str, int] = {"450k": 485_577, "epic": 865_918}

#: Standard per-chromosome PLINK files (1–22 autosomes, X, Y, MT).
CHROMOSOMES: tuple[str, ...] = (
    *(str(c) for c in range(1, 23)),
    "X", "Y", "MT",
)


def champ_arrays_for_samples(n_samples: int, platform: str = "450k") -> int:
    """Number of methylation arrays a cohort needs (one per sample).

    .. math:: n_{arrays} = n_{samples}
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples!r}")
    _check_platform(platform)
    return n_samples


def champ_chips_for_samples(n_samples: int, platform: str = "450k") -> int:
    """Number of physical BeadChips, one array per sample, per chip capacity.

    .. math:: n_{chips} = \\lceil n_{samples} / \\text{chip\\ capacity} \\rceil
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples!r}")
    _check_platform(platform)
    return math.ceil(n_samples / CHIP_CAPACITY[platform])


def champ_probe_count(platform: str = "450k") -> int:
    """Probes on the platform's array (Illumina product spec)."""
    _check_platform(platform)
    return ARRAY_PROBES[platform]


def champ_expected_failed_arrays(n_arrays: int, fail_rate: float) -> float:
    """Expected arrays failing a QC gate at a *given* fail rate (parameter, not claim).

    .. math:: n_{fail} = n_{arrays}\\, p
    """
    if n_arrays < 1:
        raise ValueError(f"n_arrays must be >= 1, got {n_arrays!r}")
    if not 0.0 <= fail_rate <= 1.0:
        raise ValueError(f"fail_rate must be in [0, 1], got {fail_rate!r}")
    return n_arrays * fail_rate


def plink_bed_size_mb(n_samples: int, n_variants: int) -> float:
    """Size of a PLINK binary ``.bed`` dataset, MB.

    .. math:: B = \\frac{n_{samples} \\times n_{variants}}{4\\,\\times\\,10^6}

    The 2-bits/sample/variant binary format packs 4 samples per byte.
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples!r}")
    if n_variants < 1:
        raise ValueError(f"n_variants must be >= 1, got {n_variants!r}")
    return n_samples * n_variants / 4.0 / 1e6


def plink_per_chr_files() -> int:
    """Number of per-chromosome PLINK files (1–22, X, Y, MT = 25)."""
    return len(CHROMOSOMES)


def plink_per_chr_bed_size_mb(n_samples: int, n_variants_chr: int) -> float:
    """Size of one chromosome's ``.bed``, given its variant count.

    Same 2-bits/sample/variant rule applied to a single chromosome's variant
    subset — lets a design size per-chromosome jobs from a split.
    """
    return plink_bed_size_mb(n_samples, n_variants_chr)


def _check_platform(platform: str) -> None:
    q = str(platform).strip().lower()
    if q not in CHIP_CAPACITY:
        raise ValueError(
            f"unknown platform {platform!r}; supported: 450k, epic"
        )


__all__ = [
    "CHIP_CAPACITY",
    "ARRAY_PROBES",
    "CHROMOSOMES",
    "champ_arrays_for_samples",
    "champ_chips_for_samples",
    "champ_probe_count",
    "champ_expected_failed_arrays",
    "plink_bed_size_mb",
    "plink_per_chr_files",
    "plink_per_chr_bed_size_mb",
]
