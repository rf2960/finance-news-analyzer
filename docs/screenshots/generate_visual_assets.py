from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUT = Path(__file__).resolve().parent

INK = "#15222e"
MUTED = "#657282"
LINE = "#d7e0e8"
PAPER = "#f7f9fb"
PANEL = "#ffffff"
TEAL = "#0f766e"
BLUE = "#2563eb"
GREEN = "#0f7a4f"
AMBER = "#b7791f"
RED = "#b42318"
PURPLE = "#6d5bd0"


def font(size: int, weight: str = "regular") -> ImageFont.FreeTypeFont:
    candidates = []
    if weight == "bold":
        candidates = [
            "C:/Windows/Fonts/segoeuib.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
        ]
    else:
        candidates = [
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


F = {
    "xs": font(16),
    "sm": font(20),
    "md": font(25),
    "lg": font(34, "bold"),
    "xl": font(48, "bold"),
    "bold_sm": font(20, "bold"),
    "bold_md": font(25, "bold"),
    "bold_lg": font(34, "bold"),
}


def text(draw: ImageDraw.ImageDraw, xy, value: str, fill=INK, fnt=None, anchor=None):
    draw.text(xy, value, fill=fill, font=fnt or F["sm"], anchor=anchor)


def wrap(draw: ImageDraw.ImageDraw, value: str, width: int, fnt) -> list[str]:
    words = value.split()
    lines: list[str] = []
    cur = ""
    for word in words:
        trial = f"{cur} {word}".strip()
        if draw.textbbox((0, 0), trial, font=fnt)[2] <= width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def rounded(draw, box, radius=18, fill=PANEL, outline=LINE, width=2):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def pill(draw, xy, label, fill, fg="#ffffff"):
    x, y = xy
    pad_x, pad_y = 14, 8
    bbox = draw.textbbox((0, 0), label, font=F["xs"])
    w, h = bbox[2] - bbox[0] + pad_x * 2, bbox[3] - bbox[1] + pad_y * 2
    draw.rounded_rectangle((x, y, x + w, y + h), radius=16, fill=fill)
    text(draw, (x + pad_x, y + pad_y - 1), label, fill=fg, fnt=F["xs"])
    return x + w


def card(draw, box, title, body, accent=TEAL, number=None):
    rounded(draw, box, radius=20)
    x1, y1, x2, y2 = box
    draw.rounded_rectangle((x1, y1, x1 + 8, y2), radius=4, fill=accent)
    if number:
        draw.ellipse((x1 + 22, y1 + 24, x1 + 58, y1 + 60), fill=accent)
        text(draw, (x1 + 40, y1 + 42), number, fill="#ffffff", fnt=F["xs"], anchor="mm")
        tx = x1 + 74
    else:
        tx = x1 + 28
    text(draw, (tx, y1 + 23), title, fnt=F["bold_sm"])
    yy = y1 + 58
    for line in wrap(draw, body, x2 - tx - 22, F["xs"])[:4]:
        text(draw, (tx, yy), line, fill=MUTED, fnt=F["xs"])
        yy += 22


def arrow(draw, start, end, color="#8b99a8", width=3):
    draw.line((start, end), fill=color, width=width)
    ang = math.atan2(end[1] - start[1], end[0] - start[0])
    for delta in (2.65, -2.65):
        p = (end[0] - 13 * math.cos(ang + delta), end[1] - 13 * math.sin(ang + delta))
        draw.line((end, p), fill=color, width=width)


def chrome_frame(draw, w=1440, h=900, title="FinSight RAG"):
    draw.rounded_rectangle((34, 26, w - 34, h - 26), radius=24, fill="#ffffff", outline=LINE, width=2)
    draw.rounded_rectangle((34, 26, w - 34, 84), radius=24, fill="#f0f4f7", outline=LINE, width=2)
    draw.rectangle((35, 58, w - 35, 84), fill="#f0f4f7")
    for i, c in enumerate(("#ff6b6b", "#f7c948", "#39b980")):
        draw.ellipse((60 + i * 24, 48, 74 + i * 24, 62), fill=c)
    text(draw, (132, 45), title, fill=MUTED, fnt=F["xs"])


def architecture():
    img = Image.new("RGB", (1800, 1050), PAPER)
    d = ImageDraw.Draw(img)
    text(d, (90, 75), "FinSight RAG Architecture", fnt=F["xl"])
    text(
        d,
        (92, 134),
        "Explainable financial-news retrieval, evidence verification, and signal synthesis",
        fill=MUTED,
        fnt=F["md"],
    )
    pill(d, (92, 184), "Lexical-first hybrid retrieval", TEAL)
    pill(d, (390, 184), "Evidence ledger", BLUE)
    pill(d, (555, 184), "Skeptical verifier", PURPLE)
    pill(d, (745, 184), "Forward-return evaluation", AMBER)

    top_y = 285
    w, h = 305, 154
    xs = [90, 460, 830, 1200]
    cards = [
        ("01", "Collect", "Yahoo Finance, Google News RSS, public financial feeds, optional Bloomberg B-PIPE.", TEAL),
        ("02", "Index", "Normalize articles, chunk text, attach ticker, company, source, date, and credibility metadata.", BLUE),
        ("03", "Retrieve", "BM25 plus TF-IDF/keyword scoring, then metadata reranking for precise evidence selection.", PURPLE),
        ("04", "Reason", "Analyst and verifier steps classify support, contradiction, risk, uncertainty, and signal strength.", AMBER),
    ]
    for x, spec in zip(xs, cards):
        card(d, (x, top_y, x + w, top_y + h), spec[1], spec[2], spec[3], spec[0])
    for x in xs[:-1]:
        arrow(d, (x + w + 18, top_y + h // 2), (x + 370 - 18, top_y + h // 2))

    rounded(d, (90, 535, 820, 895), radius=24, fill="#ffffff")
    text(d, (125, 570), "Retrieval Score Breakdown", fnt=F["bold_lg"])
    rows = [
        ("BM25 event match", 0.32, TEAL),
        ("TF-IDF / keyword overlap", 0.26, BLUE),
        ("Ticker + company match", 0.18, GREEN),
        ("Source authority", 0.10, PURPLE),
        ("Recency + financial intent", 0.14, AMBER),
    ]
    y = 635
    for label, val, col in rows:
        text(d, (130, y), label, fnt=F["sm"])
        draw_w = int(420 * val / 0.35)
        d.rounded_rectangle((390, y + 3, 760, y + 23), radius=10, fill="#e8eef3")
        d.rounded_rectangle((390, y + 3, 390 + draw_w, y + 23), radius=10, fill=col)
        text(d, (775, y - 1), f"{val:.2f}", fill=MUTED, fnt=F["xs"])
        y += 48

    rounded(d, (890, 535, 1710, 895), radius=24, fill="#ffffff")
    text(d, (925, 570), "Final Research Packet", fnt=F["bold_lg"])
    packet = [
        ("Signal", "Bullish / Bearish / Neutral with calibrated confidence"),
        ("Evidence", "Ranked snippets, citations, stance, source quality, timestamps"),
        ("Verifier", "Contradictions, stale evidence, low-source-diversity warnings"),
        ("Evaluation", "5d / 20d forward-return diagnostics and baseline comparison"),
    ]
    y = 640
    for label, body in packet:
        d.rounded_rectangle((925, y - 12, 1075, y + 28), radius=14, fill="#edf7f5")
        text(d, (948, y - 4), label, fill=TEAL, fnt=F["bold_sm"])
        text(d, (1110, y - 4), body, fill=INK, fnt=F["sm"])
        y += 60

    text(d, (90, 980), "Designed for auditability: every generated claim is tied back to retrieved evidence.", fill=MUTED, fnt=F["sm"])
    img.save(OUT / "system-architecture.png", quality=94)


def live_analysis():
    img = Image.new("RGB", (1600, 1000), "#eef3f7")
    d = ImageDraw.Draw(img)
    chrome_frame(d, 1600, 1000, "FinSight RAG - Live Analysis")
    d.rectangle((34, 84, 300, 974), fill="#17202a")
    text(d, (70, 122), "FinSight RAG", fill="#ffffff", fnt=F["bold_lg"])
    text(d, (70, 165), "Research console", fill="#aab6c2", fnt=F["sm"])
    for i, item in enumerate(["Live Analysis", "Market Scan", "Market Monitor", "Evidence Audit", "Evaluation"]):
        y = 230 + i * 58
        fill = "#22384a" if i == 0 else "#17202a"
        d.rounded_rectangle((58, y, 270, y + 42), radius=12, fill=fill)
        text(d, (78, y + 10), item, fill="#ffffff" if i == 0 else "#aab6c2", fnt=F["sm"])

    text(d, (340, 120), "Live Multi-Agent Analysis", fnt=F["xl"])
    text(d, (342, 176), "NVDA | NVIDIA Corporation | Technology", fill=MUTED, fnt=F["md"])
    pill(d, (342, 220), "Bullish", GREEN)
    pill(d, (455, 220), "62% confidence", TEAL)
    pill(d, (635, 220), "8 retrieved chunks", BLUE)

    card(d, (340, 300, 960, 540), "Signal Thesis", "The bullish thesis is supported by AI demand, raised analyst targets, and constructive data-center revenue evidence. Verifier flags valuation risk as the main counterweight.", GREEN)
    rounded(d, (995, 300, 1510, 540), radius=20, fill=PANEL)
    text(d, (1030, 330), "Market Snapshot", fnt=F["bold_lg"])
    metrics = [("Last price", "$139.20"), ("Relative strength", "+1.8%"), ("Volume vs avg", "1.42x"), ("RSI-14", "63")]
    for idx, (k, v) in enumerate(metrics):
        x = 1030 + (idx % 2) * 230
        y = 390 + (idx // 2) * 72
        rounded(d, (x, y, x + 190, y + 48), radius=12, fill="#f5f8fa")
        text(d, (x + 16, y + 8), k, fill=MUTED, fnt=F["xs"])
        text(d, (x + 16, y + 27), v, fnt=F["bold_sm"])

    rounded(d, (340, 585, 1510, 885), radius=20, fill=PANEL)
    text(d, (375, 620), "Evidence Ledger", fnt=F["bold_lg"])
    headers = ["Rank", "Stance", "Source", "Retrieval", "Credibility", "Evidence"]
    cols = [375, 455, 590, 760, 900, 1060]
    for x, h in zip(cols, headers):
        text(d, (x, 670), h, fill=MUTED, fnt=F["bold_sm"])
    rows = [
        ("1", "supports", "Reuters", "0.94", "85%", "AI data-center demand supports raised guidance."),
        ("2", "supports", "CNBC", "0.75", "78%", "Analysts lift price targets after earnings beat."),
        ("3", "challenges", "MarketWatch", "0.54", "70%", "Valuation risk may limit near-term upside."),
    ]
    y = 715
    for row in rows:
        d.line((370, y - 18, 1485, y - 18), fill=LINE, width=2)
        for x, value in zip(cols, row):
            col = GREEN if value == "supports" else RED if value == "challenges" else INK
            text(d, (x, y), value, fill=col, fnt=F["sm"])
        y += 58

    img.save(OUT / "live-analysis.png", quality=94)


def evidence_audit():
    img = Image.new("RGB", (1600, 1000), "#eef3f7")
    d = ImageDraw.Draw(img)
    chrome_frame(d, 1600, 1000, "FinSight RAG - Evidence Audit")
    text(d, (70, 120), "Evidence Audit", fnt=F["xl"])
    text(d, (72, 176), "Retrieval architecture decision + selected signal evidence", fill=MUTED, fnt=F["md"])

    rounded(d, (70, 235, 1530, 455), radius=22, fill=PANEL)
    text(d, (105, 268), "Why lexical-first hybrid retrieval?", fnt=F["bold_lg"])
    methods = [
        ("Keyword", "Fallback", "very high", "brittle phrasing"),
        ("TF-IDF", "Strong baseline", "high", "vocabulary-sensitive"),
        ("BM25", "Best sparse first stage", "high", "still lexical"),
        ("Embeddings", "Optional reranker", "medium", "ticker-wrong matches"),
        ("Cross-encoder", "Production rerank", "medium-low", "latency/cost"),
    ]
    x = 105
    for name, fit, exp, risk in methods:
        rounded(d, (x, 330, x + 260, 425), radius=16, fill="#f6f9fb")
        text(d, (x + 18, 350), name, fnt=F["bold_sm"])
        text(d, (x + 18, 379), fit, fill=TEAL, fnt=F["xs"])
        text(d, (x + 18, 401), f"Explainability: {exp}", fill=MUTED, fnt=F["xs"])
        x += 280

    rounded(d, (70, 500, 920, 900), radius=22, fill=PANEL)
    text(d, (105, 535), "Verifier Flags", fnt=F["bold_lg"])
    flags = [
        ("No major grounding issue detected", GREEN),
        ("1 item challenges final direction", AMBER),
        ("Average retrieval score: 0.71", BLUE),
        ("Sources represented: Reuters, CNBC, MarketWatch", TEAL),
    ]
    y = 600
    for label, col in flags:
        d.rounded_rectangle((105, y, 860, y + 58), radius=14, fill="#f8fafc", outline=LINE)
        d.ellipse((125, y + 18, 147, y + 40), fill=col)
        text(d, (168, y + 17), label, fnt=F["sm"])
        y += 72

    rounded(d, (970, 500, 1530, 900), radius=22, fill=PANEL)
    text(d, (1005, 535), "Grounding Mix", fnt=F["bold_lg"])
    center = (1250, 705)
    vals = [(210, GREEN, "Supports 5"), (90, RED, "Challenges 1"), (60, BLUE, "Context 2")]
    start = -90
    total = sum(v[0] for v in vals)
    for angle, col, _ in vals:
        end = start + angle / total * 360
        d.pieslice((1100, 580, 1400, 880), start, end, fill=col)
        start = end
    d.ellipse((1180, 660, 1320, 800), fill=PANEL)
    text(d, center, "8\nchunks", fnt=F["bold_md"], anchor="mm")
    y = 620
    for _, col, label in vals:
        d.rectangle((1430, y + 5, 1450, y + 25), fill=col)
        text(d, (1462, y), label, fnt=F["sm"])
        y += 42
    img.save(OUT / "evidence-audit.png", quality=94)


def market_monitor():
    img = Image.new("RGB", (1600, 1000), "#eef3f7")
    d = ImageDraw.Draw(img)
    chrome_frame(d, 1600, 1000, "FinSight RAG - Market Monitor")
    text(d, (70, 120), "Market Monitor", fnt=F["xl"])
    text(d, (72, 176), "Signal queue, conviction map, and divergence audit", fill=MUTED, fnt=F["md"])
    rounded(d, (70, 240, 930, 900), radius=22, fill=PANEL)
    text(d, (105, 275), "Signal Queue", fnt=F["bold_lg"])
    signals = [("NVDA", "Bullish", GREEN, "0.62", "AI demand + guidance"), ("MSFT", "Neutral", BLUE, "0.51", "Balanced cloud growth"), ("TSLA", "Bearish", RED, "0.58", "Margin + regulatory risk")]
    y = 345
    for t, direction, col, conf, cat in signals:
        rounded(d, (105, y, 885, y + 128), radius=16, fill="#f8fafc")
        text(d, (132, y + 24), t, fnt=F["bold_lg"])
        pill(d, (230, y + 26), direction, col)
        text(d, (132, y + 74), cat, fill=MUTED, fnt=F["sm"])
        text(d, (800, y + 28), conf, fill=col, fnt=F["bold_lg"])
        text(d, (780, y + 72), "confidence", fill=MUTED, fnt=F["xs"])
        y += 155

    rounded(d, (980, 240, 1530, 900), radius=22, fill=PANEL)
    text(d, (1015, 275), "Conviction Map", fnt=F["bold_lg"])
    d.line((1060, 760, 1460, 760), fill=LINE, width=3)
    d.line((1260, 360, 1260, 825), fill=LINE, width=3)
    points = [(1360, 470, "NVDA", GREEN, 34), (1265, 620, "MSFT", BLUE, 24), (1110, 540, "TSLA", RED, 30)]
    for x, y, label, col, r in points:
        d.ellipse((x - r, y - r, x + r, y + r), fill=col)
        text(d, (x, y - 8), label, fill="#ffffff", fnt=F["bold_sm"], anchor="mm")
    text(d, (1130, 805), "Bearish sentiment", fill=MUTED, fnt=F["xs"])
    text(d, (1320, 805), "Bullish sentiment", fill=MUTED, fnt=F["xs"])
    text(d, (1282, 365), "Higher confidence", fill=MUTED, fnt=F["xs"])
    img.save(OUT / "market-monitor.png", quality=94)


def evaluation_lab():
    img = Image.new("RGB", (1600, 1000), "#eef3f7")
    d = ImageDraw.Draw(img)
    chrome_frame(d, 1600, 1000, "FinSight RAG - Evaluation")
    text(d, (70, 120), "Evaluation Lab", fnt=F["xl"])
    text(d, (72, 176), "Directional hit rate, baseline comparison, and signal-level outcomes", fill=MUTED, fnt=F["md"])

    rounded(d, (70, 240, 1530, 470), radius=22, fill=PANEL)
    metrics = [("RAG 5d hit rate", "67%"), ("RAG 20d hit rate", "56%"), ("Avg confidence", "58%"), ("Signals evaluated", "9")]
    for idx, (label, value) in enumerate(metrics):
        x = 105 + idx * 350
        rounded(d, (x, 295, x + 290, 405), radius=18, fill="#f8fafc")
        text(d, (x + 24, 322), label, fill=MUTED, fnt=F["sm"])
        text(d, (x + 24, 354), value, fnt=F["bold_lg"])

    rounded(d, (70, 525, 1530, 900), radius=22, fill=PANEL)
    text(d, (105, 560), "Baseline Comparison", fnt=F["bold_lg"])
    groups = [("Multi-Agent RAG", 0.67, 0.56, TEAL), ("Sentiment Baseline", 0.44, 0.44, BLUE), ("Random Baseline", 0.33, 0.33, AMBER)]
    x0, y0 = 170, 800
    for idx, (label, v5, v20, col) in enumerate(groups):
        x = x0 + idx * 420
        d.rounded_rectangle((x, y0 - int(v5 * 270), x + 80, y0), radius=12, fill=col)
        d.rounded_rectangle((x + 105, y0 - int(v20 * 270), x + 185, y0), radius=12, fill="#9aa8b5")
        text(d, (x - 20, 830), label, fnt=F["sm"])
        text(d, (x + 8, y0 - int(v5 * 270) - 30), f"{v5:.0%}", fill=col, fnt=F["bold_sm"])
        text(d, (x + 118, y0 - int(v20 * 270) - 30), f"{v20:.0%}", fill=MUTED, fnt=F["bold_sm"])
    pill(d, (1220, 560), "5d", TEAL)
    pill(d, (1280, 560), "20d", "#9aa8b5")
    img.save(OUT / "evaluation-lab.png", quality=94)


def main():
    architecture()
    live_analysis()
    evidence_audit()
    market_monitor()
    evaluation_lab()


if __name__ == "__main__":
    main()
