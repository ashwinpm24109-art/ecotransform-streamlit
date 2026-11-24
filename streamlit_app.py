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

# Preparation steps shown before recycling, per model class.
PREP = {
    "cardboard": ["Remove tape, labels and plastic windows.", "Flatten the box and keep it dry."],
    "paper": ["Remove staples, spiral bindings and plastic covers.", "Keep it dry; wet paper cannot be recycled."],
    "glass": ["Rinse out food and drink residue.", "Remove metal lids; keep glass separate from ceramics and mirrors."],
    "metal": ["Rinse cans and tins to remove food.", "Crush aluminium cans if your collection allows it."],
    "plastic": ["Rinse containers and remove caps where your local rules require it.", "Check the number in the recycling triangle."],
    "trash": ["Do not put this in the recycling bin.", "Ask your local council about bulky or hazardous items."],
}

# Concrete reusable products per model class: (product, what you need, how to make it).
REUSABLE = {
    "cardboard": [
        ("Drawer organiser", "a shoebox, scissors, glue",
         ["Cut the box to the drawer's depth.", "Cut the inner dividers from thin card.", "Glue the dividers in place and let them dry."]),
        ("Plant pots", "clean boxes, a pencil, water-resistant tape",
         ["Fold a box into a tube and tape the bottom closed.", "Line the inside with a plastic bag to hold water.", "Fill with soil and plant."]),
        ("Magazine file", "a sturdy box, craft knife",
         ["Cut the front of the box down to half its height.", "Score and fold the edges so it stands upright."]),
        ("Kids' play house", "large boxes, paint, tape",
         ["Cut a door and windows into one large box.", "Tape the flaps together for a roof.", "Paint and decorate the outside."]),
    ],
    "paper": [
        ("Notebook", "used paper, a needle, thread",
         ["Stack and fold clean sheets in half.", "Stitch along the fold with thread.", "Add a cardboard cover if you want it sturdier."]),
        ("Papier-mâché bowl", "paper strips, flour and water paste, a balloon",
         ["Mix flour and water into a smooth paste.", "Dip paper strips and layer them over a blown-up balloon.", "Let it dry for two days, then pop the balloon and trim the edge."]),
        ("Gift wrap", "clean paper, stamps or paint",
         ["Press a sponge or stamp into paint.", "Print patterns on plain paper.", "Let it dry flat before wrapping."]),
        ("Seed paper cards", "shredded paper, seeds, a bowl of water",
         ["Soak shredded paper in water overnight.", "Blend it, add seeds, and press it flat on a towel.", "Let it dry; plant the card when you're done with it."]),
    ],
    "glass": [
        ("Candle holder", "a clean jar, a tealight",
         ["Wash and dry the jar completely.", "Place a tealight inside.", "Keep it away from curtains and paper."]),
        ("Hanging planter", "a bottle, jute twine, a drill with a glass bit",
         ["Mark a line of holes around the neck with a marker.", "Drill slowly with water on the glass.", "Thread the twine through and hang it."]),
        ("Storage jars", "clean jars and labels",
         ["Wash jars and lids, then dry them fully.", "Label each jar with what it holds and the date."]),
        ("Drinking glass", "a clean bottle, sandpaper for a smooth rim",
         ["Score the bottle with a glass cutter or a hot-and-cold water method.", "Smooth the cut edge with fine sandpaper."]),
    ],
    "metal": [
        ("Pencil holder", "a clean tin, paint or sticker",
         ["Remove the label and wash the tin.", "Sand lightly, then paint or cover with sticker.", "Let dry before use."]),
        ("Wall planter", "a tin can, wire, a nail for drainage",
         ["Punch a drainage hole in the base with a nail and hammer.", "Attach wire to both sides for hanging.", "Fill with soil and plant."]),
        ("Coasters", "metal lids, glue, felt pads",
         ["Smooth the rim of each lid.", "Glue a felt pad to the underside.", "Let dry before use."]),
        ("Tealight lantern", "a tin can, a nail and hammer for patterns",
         ["Fill the can with water and freeze it solid.", "Hammer dot patterns through the sides with a nail.", "Thaw, dry, and add a tealight."]),
    ],
    "plastic": [
        ("Self-watering planter", "a bottle, a cotton wick, soil",
         ["Cut the bottle in half.", "Put the top upside down into the bottom, like a funnel.", "Thread a cotton wick through the neck into the water reservoir, fill with soil and plant."]),
        ("Bird feeder", "a bottle, wooden spoons, string",
         ["Cut a window on each side of the bottle.", "Push wooden spoons through the holes to make perches.", "Hang with string and fill with seed."]),
        ("Storage containers", "clean yogurt pots and lids",
         ["Wash the pots and dry them fully.", "Label each one and stack them."]),
        ("Watering can", "a large bottle, a nail for the spout holes",
         ["Heat the nail and push holes through the cap to make a rose.", "Cut a small handle in the bottle's side.", "Fill with water and test for leaks."]),
    ],
    "trash": [
        ("Repair first", "basic tools, a repair café or a local shop",
         ["Check the item for a loose part or a blown fuse.", "Look up repair guides for that model.", "Take it to a repair café if you're unsure."]),
        ("Donate", "a clean, working item and a local charity or swap group",
         ["Check the item still works and is clean.", "Drop it at a charity shop or community swap group."]),
    ],
}

ACCENT = "#1f8a4c"
ACCENT_DARK = "#14653a"
CSS = f"""
<style>
    .block-container {{ max-width: 680px; padding-top: 2rem; }}
    .banner {{
        background: linear-gradient(135deg, {ACCENT} 0%, {ACCENT_DARK} 100%);
        color: #ffffff; border-radius: 18px; padding: 24px 28px; margin-bottom: 1.25rem;
        box-shadow: 0 8px 24px rgba(20, 101, 58, 0.25);
    }}
    .banner h1 {{ color: #ffffff; font-size: 2rem; margin: 0 0 4px 0; }}
    .banner p {{ color: #e6f7ec; margin: 0; font-size: 1rem; }}
    .badge {{
        display: inline-block; padding: 4px 12px; border-radius: 999px;
        background: rgba(31, 138, 76, 0.15); color: {ACCENT}; font-weight: 600; font-size: 0.85rem;
    }}
    .step {{ font-weight: 700; font-size: 0.8rem; letter-spacing: 0.06em; text-transform: uppercase; color: {ACCENT}; margin: 1.25rem 0 0.25rem 0; }}
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
        <div class="banner">
            <h1>♻ EcoTransform AI</h1>
            <p>Snap a photo of waste. Learn how to dispose of it and turn it into something useful.</p>
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

    st.markdown('<p class="step">Before you recycle</p>', unsafe_allow_html=True)
    st.markdown("\n".join(f"- {step}" for step in PREP[results[0][0]]))

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

    st.markdown('<p class="step">Reusable products you can make</p>', unsafe_allow_html=True)
    for name, needs, steps in REUSABLE[results[0][0]]:
        with st.expander(f"{name} · what you need: {needs}"):
            st.markdown("\n".join(f"{i}. {step}" for i, step in enumerate(steps, 1)))

    st.markdown('<p class="step">Other possible matches</p>', unsafe_allow_html=True)
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
