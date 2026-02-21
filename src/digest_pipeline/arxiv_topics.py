"""Index of arXiv subject categories for subscription enumeration.

Provides the full arXiv taxonomy so users can browse available topics
and configure their digest subscriptions from a known set.

Usage::

    from digest_pipeline.arxiv_topics import TOPICS, search_topics, list_group

    # List all CS topics
    for t in list_group("cs"):
        print(f"{t.code}: {t.name}")

    # Search by keyword
    for t in search_topics("machine learning"):
        print(f"{t.code}: {t.name}")
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArxivTopic:
    """A single arXiv subject category."""

    code: str
    name: str
    group: str


# ── Full arXiv taxonomy ────────────────────────────────────────
# Source: https://arxiv.org/category_taxonomy
TOPICS: tuple[ArxivTopic, ...] = (
    # ── Computer Science (cs) ──────────────────────────────────
    ArxivTopic("cs.AI", "Artificial Intelligence", "cs"),
    ArxivTopic("cs.AR", "Hardware Architecture", "cs"),
    ArxivTopic("cs.CC", "Computational Complexity", "cs"),
    ArxivTopic("cs.CE", "Computational Engineering, Finance, and Science", "cs"),
    ArxivTopic("cs.CG", "Computational Geometry", "cs"),
    ArxivTopic("cs.CL", "Computation and Language", "cs"),
    ArxivTopic("cs.CR", "Cryptography and Security", "cs"),
    ArxivTopic("cs.CV", "Computer Vision and Pattern Recognition", "cs"),
    ArxivTopic("cs.CY", "Computers and Society", "cs"),
    ArxivTopic("cs.DB", "Databases", "cs"),
    ArxivTopic("cs.DC", "Distributed, Parallel, and Cluster Computing", "cs"),
    ArxivTopic("cs.DL", "Digital Libraries", "cs"),
    ArxivTopic("cs.DM", "Discrete Mathematics", "cs"),
    ArxivTopic("cs.DS", "Data Structures and Algorithms", "cs"),
    ArxivTopic("cs.ET", "Emerging Technologies", "cs"),
    ArxivTopic("cs.FL", "Formal Languages and Automata Theory", "cs"),
    ArxivTopic("cs.GL", "General Literature", "cs"),
    ArxivTopic("cs.GR", "Graphics", "cs"),
    ArxivTopic("cs.GT", "Computer Science and Game Theory", "cs"),
    ArxivTopic("cs.HC", "Human-Computer Interaction", "cs"),
    ArxivTopic("cs.IR", "Information Retrieval", "cs"),
    ArxivTopic("cs.IT", "Information Theory", "cs"),
    ArxivTopic("cs.LG", "Machine Learning", "cs"),
    ArxivTopic("cs.LO", "Logic in Computer Science", "cs"),
    ArxivTopic("cs.MA", "Multiagent Systems", "cs"),
    ArxivTopic("cs.MM", "Multimedia", "cs"),
    ArxivTopic("cs.MS", "Mathematical Software", "cs"),
    ArxivTopic("cs.NA", "Numerical Analysis", "cs"),
    ArxivTopic("cs.NE", "Neural and Evolutionary Computing", "cs"),
    ArxivTopic("cs.NI", "Networking and Internet Architecture", "cs"),
    ArxivTopic("cs.OH", "Other Computer Science", "cs"),
    ArxivTopic("cs.OS", "Operating Systems", "cs"),
    ArxivTopic("cs.PF", "Performance", "cs"),
    ArxivTopic("cs.PL", "Programming Languages", "cs"),
    ArxivTopic("cs.RO", "Robotics", "cs"),
    ArxivTopic("cs.SC", "Symbolic Computation", "cs"),
    ArxivTopic("cs.SD", "Sound", "cs"),
    ArxivTopic("cs.SE", "Software Engineering", "cs"),
    ArxivTopic("cs.SI", "Social and Information Networks", "cs"),
    ArxivTopic("cs.SY", "Systems and Control", "cs"),
    # ── Economics (econ) ───────────────────────────────────────
    ArxivTopic("econ.EM", "Econometrics", "econ"),
    ArxivTopic("econ.GN", "General Economics", "econ"),
    ArxivTopic("econ.TH", "Theoretical Economics", "econ"),
    # ── Electrical Engineering and Systems Science (eess) ──────
    ArxivTopic("eess.AS", "Audio and Speech Processing", "eess"),
    ArxivTopic("eess.IV", "Image and Video Processing", "eess"),
    ArxivTopic("eess.SP", "Signal Processing", "eess"),
    ArxivTopic("eess.SY", "Systems and Control", "eess"),
    # ── Mathematics (math) ─────────────────────────────────────
    ArxivTopic("math.AC", "Commutative Algebra", "math"),
    ArxivTopic("math.AG", "Algebraic Geometry", "math"),
    ArxivTopic("math.AP", "Analysis of PDEs", "math"),
    ArxivTopic("math.AT", "Algebraic Topology", "math"),
    ArxivTopic("math.CA", "Classical Analysis and ODEs", "math"),
    ArxivTopic("math.CO", "Combinatorics", "math"),
    ArxivTopic("math.CT", "Category Theory", "math"),
    ArxivTopic("math.CV", "Complex Variables", "math"),
    ArxivTopic("math.DG", "Differential Geometry", "math"),
    ArxivTopic("math.DS", "Dynamical Systems", "math"),
    ArxivTopic("math.FA", "Functional Analysis", "math"),
    ArxivTopic("math.GM", "General Mathematics", "math"),
    ArxivTopic("math.GN", "General Topology", "math"),
    ArxivTopic("math.GR", "Group Theory", "math"),
    ArxivTopic("math.GT", "Geometric Topology", "math"),
    ArxivTopic("math.HO", "History and Overview", "math"),
    ArxivTopic("math.IT", "Information Theory", "math"),
    ArxivTopic("math.KT", "K-Theory and Homology", "math"),
    ArxivTopic("math.LO", "Logic", "math"),
    ArxivTopic("math.MG", "Metric Geometry", "math"),
    ArxivTopic("math.MP", "Mathematical Physics", "math"),
    ArxivTopic("math.NA", "Numerical Analysis", "math"),
    ArxivTopic("math.NT", "Number Theory", "math"),
    ArxivTopic("math.OA", "Operator Algebras", "math"),
    ArxivTopic("math.OC", "Optimization and Control", "math"),
    ArxivTopic("math.PR", "Probability", "math"),
    ArxivTopic("math.QA", "Quantum Algebra", "math"),
    ArxivTopic("math.RA", "Rings and Algebras", "math"),
    ArxivTopic("math.RT", "Representation Theory", "math"),
    ArxivTopic("math.SG", "Symplectic Geometry", "math"),
    ArxivTopic("math.SP", "Spectral Theory", "math"),
    ArxivTopic("math.ST", "Statistics Theory", "math"),
    # ── Physics ────────────────────────────────────────────────
    ArxivTopic("astro-ph.CO", "Cosmology and Nongalactic Astrophysics", "astro-ph"),
    ArxivTopic("astro-ph.EP", "Earth and Planetary Astrophysics", "astro-ph"),
    ArxivTopic("astro-ph.GA", "Astrophysics of Galaxies", "astro-ph"),
    ArxivTopic("astro-ph.HE", "High Energy Astrophysical Phenomena", "astro-ph"),
    ArxivTopic("astro-ph.IM", "Instrumentation and Methods for Astrophysics", "astro-ph"),
    ArxivTopic("astro-ph.SR", "Solar and Stellar Astrophysics", "astro-ph"),
    ArxivTopic("cond-mat.dis-nn", "Disordered Systems and Neural Networks", "cond-mat"),
    ArxivTopic("cond-mat.mes-hall", "Mesoscale and Nanoscale Physics", "cond-mat"),
    ArxivTopic("cond-mat.mtrl-sci", "Materials Science", "cond-mat"),
    ArxivTopic("cond-mat.other", "Other Condensed Matter", "cond-mat"),
    ArxivTopic("cond-mat.quant-gas", "Quantum Gases", "cond-mat"),
    ArxivTopic("cond-mat.soft", "Soft Condensed Matter", "cond-mat"),
    ArxivTopic("cond-mat.stat-mech", "Statistical Mechanics", "cond-mat"),
    ArxivTopic("cond-mat.str-el", "Strongly Correlated Electrons", "cond-mat"),
    ArxivTopic("cond-mat.supr-con", "Superconductivity", "cond-mat"),
    ArxivTopic("gr-qc", "General Relativity and Quantum Cosmology", "gr-qc"),
    ArxivTopic("hep-ex", "High Energy Physics - Experiment", "hep-ex"),
    ArxivTopic("hep-lat", "High Energy Physics - Lattice", "hep-lat"),
    ArxivTopic("hep-ph", "High Energy Physics - Phenomenology", "hep-ph"),
    ArxivTopic("hep-th", "High Energy Physics - Theory", "hep-th"),
    ArxivTopic("math-ph", "Mathematical Physics", "math-ph"),
    ArxivTopic("nlin.AO", "Adaptation and Self-Organizing Systems", "nlin"),
    ArxivTopic("nlin.CD", "Chaotic Dynamics", "nlin"),
    ArxivTopic("nlin.CG", "Cellular Automata and Lattice Gases", "nlin"),
    ArxivTopic("nlin.PS", "Pattern Formation and Solitons", "nlin"),
    ArxivTopic("nlin.SI", "Exactly Solvable and Integrable Systems", "nlin"),
    ArxivTopic("nucl-ex", "Nuclear Experiment", "nucl-ex"),
    ArxivTopic("nucl-th", "Nuclear Theory", "nucl-th"),
    ArxivTopic("physics.acc-ph", "Accelerator Physics", "physics"),
    ArxivTopic("physics.ao-ph", "Atmospheric and Oceanic Physics", "physics"),
    ArxivTopic("physics.app-ph", "Applied Physics", "physics"),
    ArxivTopic("physics.atm-clus", "Atomic and Molecular Clusters", "physics"),
    ArxivTopic("physics.atom-ph", "Atomic Physics", "physics"),
    ArxivTopic("physics.bio-ph", "Biological Physics", "physics"),
    ArxivTopic("physics.chem-ph", "Chemical Physics", "physics"),
    ArxivTopic("physics.class-ph", "Classical Physics", "physics"),
    ArxivTopic("physics.comp-ph", "Computational Physics", "physics"),
    ArxivTopic("physics.data-an", "Data Analysis, Statistics and Probability", "physics"),
    ArxivTopic("physics.ed-ph", "Physics Education", "physics"),
    ArxivTopic("physics.flu-dyn", "Fluid Dynamics", "physics"),
    ArxivTopic("physics.gen-ph", "General Physics", "physics"),
    ArxivTopic("physics.geo-ph", "Geophysics", "physics"),
    ArxivTopic("physics.hist-ph", "History and Philosophy of Physics", "physics"),
    ArxivTopic("physics.ins-det", "Instrumentation and Detectors", "physics"),
    ArxivTopic("physics.med-ph", "Medical Physics", "physics"),
    ArxivTopic("physics.optics", "Optics", "physics"),
    ArxivTopic("physics.plasm-ph", "Plasma Physics", "physics"),
    ArxivTopic("physics.pop-ph", "Popular Physics", "physics"),
    ArxivTopic("physics.soc-ph", "Physics and Society", "physics"),
    ArxivTopic("physics.space-ph", "Space Physics", "physics"),
    ArxivTopic("quant-ph", "Quantum Physics", "quant-ph"),
    # ── Quantitative Biology (q-bio) ──────────────────────────
    ArxivTopic("q-bio.BM", "Biomolecules", "q-bio"),
    ArxivTopic("q-bio.CB", "Cell Behavior", "q-bio"),
    ArxivTopic("q-bio.GN", "Genomics", "q-bio"),
    ArxivTopic("q-bio.MN", "Molecular Networks", "q-bio"),
    ArxivTopic("q-bio.NC", "Neurons and Cognition", "q-bio"),
    ArxivTopic("q-bio.OT", "Other Quantitative Biology", "q-bio"),
    ArxivTopic("q-bio.PE", "Populations and Evolution", "q-bio"),
    ArxivTopic("q-bio.QM", "Quantitative Methods", "q-bio"),
    ArxivTopic("q-bio.SC", "Subcellular Processes", "q-bio"),
    ArxivTopic("q-bio.TO", "Tissues and Organs", "q-bio"),
    # ── Quantitative Finance (q-fin) ──────────────────────────
    ArxivTopic("q-fin.CP", "Computational Finance", "q-fin"),
    ArxivTopic("q-fin.EC", "Economics", "q-fin"),
    ArxivTopic("q-fin.GN", "General Finance", "q-fin"),
    ArxivTopic("q-fin.MF", "Mathematical Finance", "q-fin"),
    ArxivTopic("q-fin.PM", "Portfolio Management", "q-fin"),
    ArxivTopic("q-fin.PR", "Pricing of Securities", "q-fin"),
    ArxivTopic("q-fin.RM", "Risk Management", "q-fin"),
    ArxivTopic("q-fin.ST", "Statistical Finance", "q-fin"),
    ArxivTopic("q-fin.TR", "Trading and Market Microstructure", "q-fin"),
    # ── Statistics (stat) ──────────────────────────────────────
    ArxivTopic("stat.AP", "Applications", "stat"),
    ArxivTopic("stat.CO", "Computation", "stat"),
    ArxivTopic("stat.ME", "Methodology", "stat"),
    ArxivTopic("stat.ML", "Machine Learning", "stat"),
    ArxivTopic("stat.OT", "Other Statistics", "stat"),
    ArxivTopic("stat.TH", "Statistics Theory", "stat"),
)

# Lookup dict by code for O(1) validation
_BY_CODE: dict[str, ArxivTopic] = {t.code: t for t in TOPICS}

# All distinct group names
GROUPS: tuple[str, ...] = tuple(sorted({t.group for t in TOPICS}))


def get_topic(code: str) -> ArxivTopic | None:
    """Return the ``ArxivTopic`` for *code*, or ``None`` if unknown."""
    return _BY_CODE.get(code)


def is_valid_topic(code: str) -> bool:
    """Return ``True`` if *code* is a recognised arXiv category."""
    return code in _BY_CODE


def list_group(group: str) -> list[ArxivTopic]:
    """Return all topics belonging to *group* (e.g. ``"cs"``)."""
    return [t for t in TOPICS if t.group == group]


def search_topics(query: str) -> list[ArxivTopic]:
    """Case-insensitive search across topic codes and names."""
    q = query.lower()
    return [t for t in TOPICS if q in t.code.lower() or q in t.name.lower()]


def validate_topics(codes: list[str]) -> tuple[list[str], list[str]]:
    """Split *codes* into (valid, invalid) lists."""
    valid = [c for c in codes if is_valid_topic(c)]
    invalid = [c for c in codes if not is_valid_topic(c)]
    return valid, invalid
