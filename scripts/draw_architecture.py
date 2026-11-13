"""Generate the editable infrastructure SVG; render PNG with ImageMagick."""

from pathlib import Path
from html import escape

OUT = Path(__file__).resolve().parents[1] / "docs/assets"
OUT.mkdir(parents=True, exist_ok=True)
parts = [
    """<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000">
<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#93a9bf"/></marker></defs>
<rect width="1600" height="1000" fill="#0c1422"/>
<text x="54" y="60" fill="#edf4ff" font-size="30" font-family="DejaVu Sans" font-weight="bold">DISTRIBUTED LoRA · SYSTEM ARCHITECTURE</text>
<text x="54" y="92" fill="#91a7be" font-size="17" font-family="DejaVu Sans">Document QA • data contracts → adapter training → evaluation → inference</text>"""
]


def group(x, y, w, h, title, color):
    parts.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" fill="{color}" fill-opacity=".06" stroke="{color}" stroke-opacity=".65"/><text x="{x + 22}" y="{y + 32}" fill="{color}" font-size="19" font-family="DejaVu Sans" font-weight="bold">{escape(title)}</text>'
    )


def node(x, y, w, title, sub, color):
    parts.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="88" rx="10" fill="#162335" stroke="{color}"/><rect x="{x}" y="{y + 15}" width="4" height="58" rx="2" fill="{color}"/><text x="{x + w / 2}" y="{y + 34}" text-anchor="middle" fill="#f0f5ff" font-size="18" font-family="DejaVu Sans" font-weight="bold">{escape(title)}</text><text x="{x + w / 2}" y="{y + 62}" text-anchor="middle" fill="#aabbd0" font-size="14" font-family="DejaVu Sans">{escape(sub)}</text>'
    )


def arrow(points, label="", x=0, y=0):
    pts = " ".join(f"{a},{b}" for a, b in points)
    parts.append(
        f'<polyline points="{pts}" fill="none" stroke="#93a9bf" stroke-width="2" marker-end="url(#arrow)"/>'
    )
    if label:
        parts.append(
            f'<text x="{x}" y="{y}" fill="#acbfd1" font-size="13" font-family="DejaVu Sans">{escape(label)}</text>'
        )


cyan = "#59d8c5"
blue = "#71b7ff"
gold = "#efbf68"
purple = "#bc9aff"
group(40, 125, 1520, 190, "01   DATA PREPARATION", cyan)
for x, t, s in [
    (70, "JSONL documents", "Context · question · answer"),
    (420, "Validate & quarantine", "Schema · duplicates · conflicts"),
    (770, "Group & split", "Document IDs + context hashes"),
    (1120, "Tokenize & mask", "Answer-only labels · preserve EOS"),
]:
    node(x, 188, 300, t, s, cyan)
for x in [370, 720, 1070]:
    arrow([(x, 232), (x + 50, 232)])
group(40, 350, 980, 280, "02   ADAPTER TRAINING", blue)
node(70, 410, 260, "Frozen base + LoRA", "PEFT · optional NF4", blue)
node(390, 410, 260, "Accelerate workers", "Single device / DDP", blue)
node(710, 410, 270, "Optimizer updates", "Accumulation · scheduler", blue)
arrow([(330, 454), (390, 454)])
arrow([(650, 454), (710, 454)])
node(390, 524, 260, "Recovery checkpoint", "Adapter · RNG · optimizer", blue)
arrow([(845, 498), (845, 568), (650, 568)], "Save", 750, 552)
arrow([(390, 568), (360, 568), (360, 484), (390, 484)], "Resume", 283, 555)
arrow([(1270, 276), (1270, 330), (200, 330), (200, 410)], "Train split", 550, 344)
group(1060, 350, 500, 280, "03   EVALUATE & SELECT", gold)
node(1100, 410, 420, "Base vs. adapter", "Answer NLL / PPL · exact match · F1", gold)
node(1100, 524, 420, "Rank experiments", "Validation selection · test held out", gold)
arrow([(1020, 454), (1100, 454)])
arrow([(1310, 498), (1310, 524)])
group(40, 665, 980, 210, "04   INFERENCE DELIVERY", purple)
node(70, 730, 260, "Adapter artifact", "Weights · tokenizer · run manifest", purple)
node(390, 730, 260, "FastAPI engine", "Readiness · optional bearer auth", purple)
node(710, 730, 270, "Generation / export", "Token limits · safe model merge", purple)
arrow([(330, 774), (390, 774)])
arrow([(650, 774), (710, 774)])
arrow([(200, 498), (200, 730)], "Export", 212, 654)
group(1060, 665, 500, 210, "05   REPRODUCIBILITY", cyan)
node(1100, 730, 420, "GitHub Actions", "Lint · contracts · CPU + 2-worker tests", cyan)
arrow([(1310, 630), (1310, 665)], "JSON reports", 1320, 653)
parts.append("""<text x="54" y="922" fill="#91a7be" font-size="16" font-family="DejaVu Sans">Verified: offline CPU training, resume equivalence, two-worker Gloo, API generation and adapter merging.</text>
<text x="54" y="952" fill="#6f879f" font-size="14" font-family="DejaVu Sans">CUDA / QLoRA paths require compatible GPU hardware. Synthetic tiny-model fixtures verify execution, not task quality.</text></svg>""")
(OUT / "architecture.svg").write_text("\n".join(parts))
