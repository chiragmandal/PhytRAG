"""
Generates docs/architecture.png — the PhytRAG architecture diagram.
Run from the project root:
    python docs/generate_architecture.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ── Palette ───────────────────────────────────────────────────────────────────
BG       = "#0d1117"
PANEL_BG = "#161b22"
BORDER   = "#30363d"

C_USER   = "#388bfd"
C_API    = "#3fb950"
C_EMBED  = "#a371f7"
C_VECTOR = "#f78166"
C_LLM    = "#ffa657"
C_OBS    = "#79c0ff"
C_INGEST = "#56d364"
C_PMC    = "#8b949e"

TEXT_PRI = "#e6edf3"
TEXT_SEC = "#8b949e"
ARROW    = "#484f58"

# ── Canvas ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(20, 11))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 20)
ax.set_ylim(0, 11)
ax.axis("off")


# ── Helpers ───────────────────────────────────────────────────────────────────

def box(x, y, w, h, color, label, sublabel=None, radius=0.32):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=1.6, edgecolor=color,
        facecolor=color, alpha=0.13, zorder=2,
    )
    ax.add_patch(patch)
    FancyBboxPatch_border = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=1.6, edgecolor=color,
        facecolor="none", zorder=3,
    )
    ax.add_patch(FancyBboxPatch_border)
    cy = y + h / 2 + (0.13 if sublabel else 0)
    ax.text(x + w / 2, cy, label,
            ha="center", va="center", color=TEXT_PRI,
            fontsize=9.5, fontweight="bold", zorder=4)
    if sublabel:
        ax.text(x + w / 2, y + h / 2 - 0.2, sublabel,
                ha="center", va="center", color=TEXT_SEC,
                fontsize=7.5, zorder=4)


def panel(x, y, w, h, title, color):
    FancyBboxPatch_bg = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0,rounding_size=0.5",
        linewidth=1, edgecolor=BORDER,
        facecolor=PANEL_BG, zorder=1,
    )
    ax.add_patch(FancyBboxPatch_bg)
    ax.text(x + 0.3, y + h - 0.3, title,
            ha="left", va="top", color=color,
            fontsize=8.5, fontweight="bold", fontstyle="italic", zorder=4)


def arr(x1, y1, x2, y2, color=ARROW, lw=1.6, rad=0.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle="-|>", color=color, lw=lw,
                    mutation_scale=12,
                    connectionstyle=f"arc3,rad={rad}"),
                zorder=5)


def alabel(x, y, text):
    ax.text(x, y, text, ha="center", va="center",
            color=TEXT_SEC, fontsize=7.2, zorder=6,
            bbox=dict(boxstyle="round,pad=0.18",
                      facecolor=BG, edgecolor="none", alpha=0.9))


# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(10, 10.6, "PhytRAG — Architecture",
        ha="center", va="center", color=TEXT_PRI,
        fontsize=16, fontweight="bold", zorder=6)
ax.text(10, 10.22, "Production RAG service · open-access plant biology literature",
        ha="center", va="center", color=TEXT_SEC, fontsize=9.5, zorder=6)

# ── Panels ────────────────────────────────────────────────────────────────────
panel(0.3, 5.1, 12.5, 4.7,  "Query pipeline  (runtime)",       C_API)
panel(0.3, 0.3, 12.5, 4.4,  "Ingestion pipeline  (one-shot)",  C_INGEST)
panel(13.1, 0.3, 6.5, 9.5,  "Observability stack",             C_OBS)

# ── QUERY PIPELINE ────────────────────────────────────────────────────────────
#   User(0.7,7.85)  →  FastAPI(3.2,7.85)  →  Response(10.2,7.85)
#                                ↓↑ embed         ↓↑ retrieve       ↓ generate
#                          Embedder(6.1,8.8)  Qdrant(6.1,7.2)  Ollama(6.1,5.55)

box(0.7,  7.85, 2.1, 0.9, C_USER,   "User",            "Streamlit UI  :8501")
box(3.2,  7.85, 2.3, 0.9, C_API,    "FastAPI",         "/query  :8000")
box(6.1,  8.8,  2.5, 0.9, C_EMBED,  "Embedder",        "all-MiniLM-L6-v2")
box(6.1,  7.2,  2.5, 0.9, C_VECTOR, "Qdrant",          "cosine · top-5  :6333")
box(6.1,  5.55, 2.5, 0.9, C_LLM,    "Ollama",          "llama3.2:3b  :11434")
box(9.5,  7.85, 2.5, 0.9, C_USER,   "Response",        "JSON + citations")

# User → FastAPI
arr(2.8, 8.3, 3.2, 8.3, C_USER, 2.0)
alabel(3.0, 8.52, "question")

# FastAPI → Embedder (up-right)
arr(4.35, 8.75, 6.1, 9.08, C_EMBED, 1.6)
alabel(5.0, 9.12, "① embed query")

# FastAPI → Qdrant (right)
arr(5.5, 8.1, 6.1, 7.65, C_VECTOR, 1.6)
alabel(5.65, 7.75, "② retrieve")

# FastAPI → Ollama (down-right)
arr(4.35, 7.95, 6.1, 6.0, C_LLM, 1.6)
alabel(4.9, 6.75, "④ generate")

# Embedder → Qdrant (query vector, down)
arr(7.35, 8.8, 7.35, 8.1, C_EMBED, 1.6)
alabel(7.95, 8.45, "query vector")

# Qdrant → FastAPI (chunks back)
arr(6.1, 7.65, 5.5, 8.1, C_VECTOR, 1.6)
alabel(5.65, 8.0, "③ chunks")

# Ollama → FastAPI (answer back)
arr(6.1, 6.0, 4.35, 7.95, C_LLM, 1.6)
alabel(4.9, 7.2, "⑤ answer")

# FastAPI → Response
arr(5.5, 8.3, 9.5, 8.3, C_API, 2.0)
alabel(7.5, 8.52, "structured response")

# Metrics side channel
arr(12.0, 8.3, 13.1, 5.55, C_OBS, 1.6, rad=-0.15)
alabel(12.85, 7.2, "⑥ metrics")

# ── INGESTION PIPELINE ───────────────────────────────────────────────────────
box(0.7,  1.9, 2.2, 0.9, C_PMC,    "NCBI PMC OA",        "open-access XML")
box(3.4,  1.9, 2.4, 0.9, C_INGEST, "download_corpus",    "~50 papers")
box(6.3,  1.9, 2.5, 0.9, C_INGEST, "chunk_and_embed",    "400-word windows")
box(9.6,  1.9, 2.3, 0.9, C_VECTOR, "Qdrant",             "987 vectors")

arr(2.9,  2.35, 3.4,  2.35, C_INGEST, 1.6)
arr(5.8,  2.35, 6.3,  2.35, C_INGEST, 1.6)
arr(8.8,  2.35, 9.6,  2.35, C_VECTOR, 1.6)
alabel(9.2, 2.6, "index")

# Dashed line: ingestion Qdrant shares volume with query Qdrant
ax.plot([10.75, 10.75], [2.8, 7.25], color=BORDER,
        lw=1.2, linestyle="--", zorder=2)
ax.text(10.9, 5.05, "shared\nDocker volume",
        ha="left", va="center", color=TEXT_SEC,
        fontsize=6.8, rotation=90, zorder=6)

# ── OBSERVABILITY STACK ──────────────────────────────────────────────────────
box(13.4, 8.1,  2.6, 0.9, C_OBS,    "Prometheus",   "scrapes /metrics  :9091")
box(13.4, 6.5,  2.6, 0.9, C_OBS,    "Grafana",      "dashboards  :3001")
box(13.4, 4.9,  2.6, 0.9, C_INGEST, "MLflow",       "eval tracking  :5002")
box(16.4, 8.1,  2.6, 0.9, C_VECTOR, "Qdrant UI",    "vector browser  :6333")
box(16.4, 6.5,  2.6, 0.9, C_API,    "API /docs",    "Swagger  :8000")
box(16.4, 4.9,  2.6, 0.9, C_USER,   "Streamlit UI", "http://localhost:8501")

arr(16.0, 8.55, 16.4, 8.55, C_OBS,    1.4)
arr(16.0, 7.0,  16.4, 7.0,  C_API,    1.4)
arr(16.0, 5.35, 16.4, 5.35, C_USER,   1.4)

arr(14.7, 8.1, 14.7, 7.4, C_OBS, 1.6)
alabel(15.25, 7.75, "feeds")

# ── Legend ────────────────────────────────────────────────────────────────────
items = [
    mpatches.Patch(facecolor=C_USER,   label="UI / User"),
    mpatches.Patch(facecolor=C_API,    label="FastAPI"),
    mpatches.Patch(facecolor=C_EMBED,  label="Embeddings"),
    mpatches.Patch(facecolor=C_VECTOR, label="Qdrant"),
    mpatches.Patch(facecolor=C_LLM,    label="Ollama LLM"),
    mpatches.Patch(facecolor=C_OBS,    label="Observability"),
    mpatches.Patch(facecolor=C_INGEST, label="Ingestion / MLflow"),
    mpatches.Patch(facecolor=C_PMC,    label="External data"),
]
leg = ax.legend(
    handles=items, loc="lower center",
    bbox_to_anchor=(0.5, -0.01),
    ncol=8, framealpha=0.0,
    labelcolor=TEXT_SEC, fontsize=7.8,
    handlelength=1.2, handleheight=0.85,
    borderpad=0.4, columnspacing=1.1,
)
leg.get_frame().set_facecolor(PANEL_BG)
leg.get_frame().set_edgecolor(BORDER)

plt.tight_layout(pad=0.2)
out = "docs/architecture.png"
plt.savefig(out, dpi=180, bbox_inches="tight",
            facecolor=BG, edgecolor="none")
print(f"Saved {out}")
