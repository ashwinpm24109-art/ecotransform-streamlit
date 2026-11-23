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
        "reuse": ["Use cardboard as drawer dividers, seed-starter pots or mulch layers."],
        "ideas": ["Molded-pulp trays and egg-style cartons", "Corrugated-board furniture panels"],
        "co2": 0.9, "value": 8,
    },
    "paper": {
        "label": "Paper",
        "bin": "Paper: keep dry; remove staples and plastic windows.",
        "reuse": ["Shred office paper for compost bins or packing material."],
        "ideas": ["Handmade notebooks and gift wrap from paper scraps"],
        "co2": 0.9, "value": 8,
    },
    "glass": {
        "label": "Glass",
        "bin": "Glass: rinse; lids and ceramics usually go elsewhere.",
        "reuse": ["Reuse jars for storage or turn bottles into hanging planters."],
        "ideas": ["Crushed-glass aggregate for paving and terrazzo", "Upcycled glass tumblers and vases"],
        "co2": 0.3, "value": 4,
    },
    "metal": {
        "label": "Metal",
        "bin": "Metal cans: rinse; aluminium and steel are both widely recycled.",
        "reuse": ["Use tins as pencil holders, planters or candle moulds."],
        "ideas": ["Aluminium-can wind chimes and lamp shades", "Recycled-steel shelving brackets"],
        "co2": 2.5, "value": 100,
    },
    "plastic": {
        "label": "Plastic",
        "bin": "Plastics: check the resin code (#1 PET and #2 HDPE are most accepted).",
        "reuse": ["Use bottles as planters or self-watering seed starters."],
        "ideas": ["Plastic-bottle yarn for tote bags and hats", "Recycled-plastic bricks for garden edging"],
        "co2": 1.8, "value": 35,
    },
    "trash": {
        "label": "General waste",
        "bin": "Not recyclable: put in the general waste bin.",
        "reuse": ["Check whether the item can be repaired or donated before throwing it away."],
        "ideas": [],
        "co2": 0.0, "value": 0,
    },
}

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


def main():
    st.set_page_config(page_title="EcoTransform AI", page_icon="♻", layout="centered")
    st.title("♻ EcoTransform AI")
    st.write("Upload a photo of a waste item to see how to dispose of it, reuse it, and what it is worth.")

    model, classes = load_model()

    weight = st.number_input("Approximate weight (kg)", min_value=0.01, value=0.5, step=0.1)
    uploaded = st.file_uploader("Photo of the item", type=["jpg", "jpeg", "png", "webp"])

    if uploaded is None:
        st.info("Choose a photo to start.")
        return

    image = Image.open(uploaded)
    st.image(image, caption="Your item", use_container_width=True)

    results = predict(image, model, classes)
    best_name, best_prob = results[0]
    info = INFO[best_name]

    st.subheader(f"Identified as: {info['label']}")
    st.progress(best_prob, text=f"Confidence {best_prob:.0%}")
    if best_prob < 0.6:
        st.warning("Low confidence. Check the category below before acting on it.")

    st.markdown(f"**How to dispose:** {info['bin']}")

    if best_name != "trash":
        co2 = info["co2"] * weight
        value = info["value"] * weight
        col1, col2 = st.columns(2)
        col1.metric("CO₂e avoided", f"{co2:.2f} kg")
        col2.metric("Potential value", f"₹{value:,.0f}")
        st.caption("Estimates for the entered weight. Indicative, not certified.")

        st.markdown("**Reuse first**")
        for tip in info["reuse"]:
            st.write(f"- {tip}")
        if info["ideas"]:
            st.markdown("**Product ideas**")
            for idea in info["ideas"]:
                st.write(f"- {idea}")

    with st.expander("Other possibilities"):
        for name, prob in results[1:]:
            st.write(f"{INFO[name]['label']}: {prob:.0%}")


if __name__ == "__main__":
    main()
