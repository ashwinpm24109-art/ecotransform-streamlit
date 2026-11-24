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
    .block-container {{ max-width: 640px; padding-top: 2.5rem; }}
    .title {{ text-align: center; margin-bottom: 0.25rem; }}
    .title h1 {{ font-size: 2rem; margin: 0; }}
    .subtitle {{ text-align: center; opacity: 0.75; margin-bottom: 1.5rem; }}
    .badge {{
        display: inline-block; padding: 4px 12px; border-radius: 999px;
        background: rgba(31, 138, 76, 0.15); color: {ACCENT}; font-weight: 600; font-size: 0.85rem;
    }}
    .step {{ font-weight: 600; font-size: 0.8rem; letter-spacing: 0.06em; text-transform: uppercase; opacity: 0.7; margin: 1rem 0 0.25rem 0; }}
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
        <div class="title"><h1>♻ EcoTransform AI</h1></div>
        <p class="subtitle">Snap a photo of waste. Learn how to dispose of it and reuse it.</p>
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
    st.set_page_config(page_title="EcoTransform AI", page_icon="♻", layout="centered")
    show_header()

    model, classes = load_model()

    st.markdown('<p class="step">1 · Add a photo</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Photo of the item", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed"
    )
    weight = st.number_input("Approximate weight (kg)", min_value=0.01, value=0.5, step=0.1)

    if uploaded is None:
        st.info("Upload a photo to see the result.")
        return

    image = Image.open(uploaded)
    st.image(image, use_container_width=True)

    results = predict(image, model, classes)
    best_name, best_prob = results[0]
    info = INFO[best_name]

    st.markdown('<p class="step">2 · Result</p>', unsafe_allow_html=True)
    show_result(info, best_prob, info["co2"] * weight, info["value"] * weight, results, weight)


if __name__ == "__main__":
    main()
