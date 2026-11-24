"""EcoTransform AI - Streamlit front end for the trained waste classifier.

Run:
    streamlit run streamlit_app.py
"""

from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

MODEL_PATH = Path(__file__).parent / "model" / "waste_classifier.pt"

# Model class -> EcoTransform knowledge-base entry (rupee values per kg, CO2 kg per kg).
INFO = {
    "cardboard": {
        "label": "Cardboard",
        "bin": "Paper / cardboard: keep dry and flattened; remove tape and plastic windows.",
        "reuse": [
            "Use cardboard as drawer dividers, seed-starter pots or mulch layers.",
            "Make a storage box or bookshelf organiser from sturdy boxes.",
            "Use flat sheets as a drawing or painting surface.",
        ],
        "ideas": [
            "Molded-pulp trays and egg-style cartons",
            "Corrugated-board furniture panels",
            "Flat-pack cardboard shelves and stools",
        ],
        "co2": 0.9, "value": 8,
    },
    "paper": {
        "label": "Paper",
        "bin": "Paper: keep dry; remove staples and plastic windows.",
        "reuse": [
            "Shred office paper for compost bins or packing material.",
            "Print on the back of used paper for notes and drafts.",
            "Make papier-mâché for craft projects.",
        ],
        "ideas": [
            "Handmade notebooks and gift wrap from paper scraps",
            "Recycled stationery and greeting cards",
            "Seed paper that can be planted after use",
        ],
        "co2": 0.9, "value": 8,
    },
    "glass": {
        "label": "Glass",
        "bin": "Glass: rinse; lids and ceramics usually go elsewhere.",
        "reuse": [
            "Reuse jars for storage or turn bottles into hanging planters.",
            "Use glass jars as drinking glasses or candle holders.",
            "Make a bird feeder or lantern from a jar.",
        ],
        "ideas": ["Crushed-glass aggregate for paving and terrazzo", "Upcycled glass tumblers and vases"],
        "co2": 0.3, "value": 4,
    },
    "metal": {
        "label": "Metal",
        "bin": "Metal cans: rinse; aluminium and steel are both widely recycled.",
        "reuse": [
            "Use tins as pencil holders, planters or candle moulds.",
            "Turn metal lids into coasters or magnets.",
            "Plant herbs in old metal containers with drainage holes.",
        ],
        "ideas": ["Aluminium-can wind chimes and lamp shades", "Recycled-steel shelving brackets"],
        "co2": 2.5, "value": 100,
    },
    "plastic": {
        "label": "Plastic",
        "bin": "Plastics: check the resin code (#1 PET and #2 HDPE are most accepted).",
        "reuse": [
            "Use bottles as planters or self-watering seed starters.",
            "Make a bird feeder or watering can from a bottle.",
            "Store small items in clean containers and yogurt pots.",
        ],
        "ideas": ["Plastic-bottle yarn for tote bags and hats", "Recycled-plastic bricks for garden edging"],
        "co2": 1.8, "value": 35,
    },
    "trash": {
        "label": "General waste",
        "bin": "Not recyclable: put in the general waste bin.",
        "reuse": [
            "Check whether the item can be repaired or donated before throwing it away.",
            "Mixed or soiled items usually cannot be recycled, so check your local rules.",
        ],
        "ideas": ["Repair cafés and community swap groups can extend the life of many items."],
        "co2": 0.0, "value": 0,
    },
}

ACCENT = "#1f8a4c"
ACCENT_DARK = "#14653a"
CSS = f"""
<style>
    .block-container {{ max-width: 1100px; padding-top: 2rem; }}
    .hero {{
        background: linear-gradient(135deg, {ACCENT} 0%, {ACCENT_DARK} 100%);
        color: #ffffff; border-radius: 18px; padding: 28px 32px; margin-bottom: 24px;
    }}
    .hero h1 {{ color: #ffffff; margin: 8px 0 10px 0; font-size: 2.1rem; line-height: 1.15; }}
    .hero .accent {{ color: #b7f5d0; }}
    .hero p {{ margin: 0; opacity: 0.92; font-size: 1.02rem; }}
    .pill {{
        display: inline-block; padding: 4px 12px; border-radius: 999px;
        background: rgba(255, 255, 255, 0.18); font-size: 0.85rem; font-weight: 600;
    }}
    .card {{
        border: 1px solid rgba(128, 128, 128, 0.25); border-radius: 16px;
        padding: 20px 22px; margin-bottom: 16px;
    }}
    .badge {{
        display: inline-block; padding: 4px 12px; border-radius: 999px;
        background: rgba(31, 138, 76, 0.15); color: {ACCENT}; font-weight: 600; font-size: 0.85rem;
    }}
    .step {{ font-weight: 600; font-size: 0.8rem; letter-spacing: 0.06em; text-transform: uppercase; opacity: 0.7; }}
    .tip li {{ margin-bottom: 4px; }}
</style>
"""

TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    classes = checkpoint["classes"]
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(classes))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, classes


def predict(image: Image.Image, model, classes):
    x = TF(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    top = torch.topk(probs, k=3)
    return [(classes[i], float(p)) for p, i in zip(top.values, top.indices)]


def show_header():
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero">
            <span class="pill">Turning today's trash into tomorrow's cash</span>
            <h1>AI-Powered Waste-to-Wealth <span class="accent">Intelligence Platform</span></h1>
            <p>Snap a photo of discarded materials. The AI classifies the material, shows how to
            dispose of it, and suggests DIY upcycling projects with estimated savings.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_result(info, best_prob, co2, value, results, weight):
    st.markdown(f'<span class="badge">{info["label"]} · {best_prob:.0%} confident</span>', unsafe_allow_html=True)
    st.markdown(f"### {info['label']}")
    if best_prob < 0.6:
        st.warning("Low confidence. Check the category before acting on it.")

    st.markdown('<p class="step">How to dispose</p>', unsafe_allow_html=True)
    st.write(info["bin"])

    if info["label"] != "General waste":
        col1, col2 = st.columns(2)
        col1.metric("CO₂e avoided", f"{co2:.2f} kg", help=f"For {weight:g} kg of this material")
        col2.metric("Potential value", f"₹{value:,.0f}", help=f"For {weight:g} kg of this material")
        st.caption("Indicative estimates, not certified figures.")

    tiers = [
        ("Beginner", info["reuse"][:1]),
        ("Intermediate", info["reuse"][1:]),
        ("Artisan", info["ideas"]),
    ]
    st.markdown('<p class="step">DIY upcycling ideas</p>', unsafe_allow_html=True)
    for level, items in tiers:
        if items:
            st.markdown(f"**{level}**")
            st.markdown("\n".join(f"- {item}" for item in items))

    with st.expander("Other possibilities"):
        for name, prob in results[1:]:
            st.write(f"{INFO[name]['label']}: {prob:.0%}")


def main():
    st.set_page_config(page_title="EcoTransform AI", page_icon="♻", layout="wide")
    show_header()

    model, classes = load_model()

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="step">Step 1</p>', unsafe_allow_html=True)
        st.markdown("#### Upload a photo")
        uploaded = st.file_uploader(
            "Photo of the item", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed"
        )
        weight = st.number_input("Approximate weight (kg)", min_value=0.01, value=0.5, step=0.1)
        st.markdown("</div>", unsafe_allow_html=True)

        if uploaded is not None:
            st.image(Image.open(uploaded), use_container_width=True)

    with right:
        if uploaded is None:
            st.markdown(
                '<div class="card"><p class="step">Step 2</p><h4>Your result appears here</h4>'
                "<p>Upload a photo on the left to see the material, how to dispose of it, "
                "and ways to reuse it.</p></div>",
                unsafe_allow_html=True,
            )
            return

        image = Image.open(uploaded)
        results = predict(image, model, classes)
        best_name, best_prob = results[0]
        info = INFO[best_name]
        co2 = info["co2"] * weight
        value = info["value"] * weight

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="step">Step 2</p>', unsafe_allow_html=True)
        show_result(info, best_prob, co2, value, results, weight)
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
