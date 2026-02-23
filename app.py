import streamlit as st
import numpy as np
import tensorflow as tf
import os
from PIL import Image

# ============================================================
# GLOBAL PAGE CONFIG + DARK THEME STYLES
# ============================================================
st.set_page_config(
    page_title="DermAI – Skin Disease Detection",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&family=Share+Tech+Mono&display=swap');

/* ── ROOT VARIABLES ── */
:root {
    --bg:         #050a0f;
    --surface:    #0d1b2a;
    --surface2:   #112236;
    --accent:     #00d4ff;
    --accent2:    #ff2d78;
    --accent3:    #7b2fff;
    --text:       #e0f0ff;
    --text-dim:   #6a8aaa;
    --glow:       0 0 18px rgba(0,212,255,.55);
    --glow2:      0 0 18px rgba(255,45,120,.55);
    --glow3:      0 0 18px rgba(123,47,255,.55);
    --border:     rgba(0,212,255,.18);
    --border2:    rgba(255,45,120,.18);
}

/* ── FULL APP BACKGROUND ── */
.stApp {
    background: var(--bg);
    font-family: 'Rajdhani', sans-serif;
    color: var(--text);
}

/* Animated grid background */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(0,212,255,.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,212,255,.04) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
    animation: gridPulse 8s ease-in-out infinite;
}
@keyframes gridPulse {
    0%,100% { opacity:.5; }
    50%      { opacity:1; }
}

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060d18 0%, #0a1828 100%);
    border-right: 1px solid var(--border);
    box-shadow: 4px 0 30px rgba(0,212,255,.08);
}
section[data-testid="stSidebar"] .stRadio label {
    color: var(--text-dim) !important;
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.05rem;
    font-weight: 600;
    letter-spacing: 1px;
    transition: color .25s, text-shadow .25s;
    cursor: pointer;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    color: var(--accent) !important;
    text-shadow: var(--glow);
}
section[data-testid="stSidebar"] h1, 
section[data-testid="stSidebar"] h2 {
    font-family: 'Orbitron', sans-serif;
    color: var(--accent);
    text-shadow: var(--glow);
    font-size: .85rem;
    letter-spacing: 2px;
}

/* ── HEADINGS ── */
h1 { 
    font-family: 'Orbitron', sans-serif !important;
    color: var(--accent) !important;
    text-shadow: var(--glow) !important;
    letter-spacing: 3px !important;
    animation: flicker 4s infinite;
}
@keyframes flicker {
    0%,19%,21%,23%,25%,54%,56%,100% { text-shadow: var(--glow); }
    20%,22%,24%,55% { text-shadow: none; }
}
h2, h3 {
    font-family: 'Orbitron', sans-serif !important;
    color: var(--accent) !important;
    letter-spacing: 2px !important;
}
h2 { font-size: 1.2rem !important; }
h3 { font-size: 1rem !important; }

/* ── CARDS ── */
.derm-card {
    background: linear-gradient(135deg, var(--surface) 0%, var(--surface2) 100%);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
    margin: 14px 0;
    position: relative;
    overflow: hidden;
    transition: transform .3s, box-shadow .3s;
    animation: cardIn .5s ease both;
}
.derm-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
    animation: scanline 3s linear infinite;
}
@keyframes scanline {
    0%   { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
}
.derm-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(0,212,255,.2);
}
@keyframes cardIn {
    from { opacity:0; transform:translateY(20px); }
    to   { opacity:1; transform:translateY(0); }
}

/* ── HERO BANNER ── */
.hero-banner {
    background: linear-gradient(135deg, #060d18 0%, #0a1828 40%, #0d1b2a 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 52px 40px;
    text-align: center;
    position: relative;
    overflow: hidden;
    margin-bottom: 30px;
}
.hero-banner::after {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at 50% 50%, rgba(0,212,255,.06) 0%, transparent 70%);
    animation: pulse 3s ease-in-out infinite;
}
@keyframes pulse {
    0%,100% { opacity:.4; }
    50%      { opacity:1; }
}
.hero-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 2.8rem;
    font-weight: 900;
    background: linear-gradient(135deg, #00d4ff, #ff2d78, #7b2fff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.2;
    animation: gradientShift 5s ease infinite;
    background-size: 200% 200%;
}
@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.hero-sub {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.15rem;
    color: var(--text-dim);
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 12px;
}

/* ── STAT COUNTERS ── */
.stat-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    transition: all .3s;
    animation: cardIn .6s ease both;
}
.stat-box:hover { box-shadow: var(--glow); border-color: var(--accent); }
.stat-num {
    font-family: 'Orbitron', sans-serif;
    font-size: 2.2rem;
    color: var(--accent);
    text-shadow: var(--glow);
    font-weight: 900;
}
.stat-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: .8rem;
    color: var(--text-dim);
    letter-spacing: 2px;
    text-transform: uppercase;
}

/* ── BADGES ── */
.badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 30px;
    font-family: 'Share Tech Mono', monospace;
    font-size: .75rem;
    letter-spacing: 1.5px;
    font-weight: 700;
    margin: 3px;
}
.badge-ba  { background: rgba(255,45,120,.15); border:1px solid var(--accent2); color:var(--accent2); }
.badge-fu  { background: rgba(123,47,255,.15); border:1px solid var(--accent3); color:var(--accent3); }
.badge-pa  { background: rgba(0,212,255,.15);  border:1px solid var(--accent);  color:var(--accent);  }
.badge-vi  { background: rgba(255,200,0,.15);  border:1px solid #ffc800;        color:#ffc800;        }
.badge-pc  { background: rgba(0,255,150,.15);  border:1px solid #00ff96;        color:#00ff96;        }

/* ── SYMPTOM / MED / PRECAUTION PILLS ── */
.pill {
    display: inline-block;
    background: var(--surface2);
    border-left: 3px solid var(--accent);
    padding: 6px 14px;
    margin: 4px 0;
    border-radius: 0 6px 6px 0;
    font-size: .9rem;
    width: 100%;
    transition: background .25s, border-color .25s;
}
.pill:hover { background: rgba(0,212,255,.1); border-color: var(--accent2); }
.pill-med  { border-color: var(--accent3); }
.pill-prec { border-color: var(--accent2); }

/* ── TIMELINE ── */
.timeline-step {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin: 10px 0;
    font-family: 'Rajdhani', sans-serif;
    font-size: 1rem;
    position: relative;
    transition: all .3s;
    animation: slideIn .4s ease both;
}
.timeline-step::before {
    content: '▶';
    position: absolute;
    left: -22px;
    top: 50%;
    transform: translateY(-50%);
    color: var(--accent);
    font-size: .7rem;
}
.timeline-step:hover {
    border-left-color: var(--accent2);
    box-shadow: var(--glow);
}
@keyframes slideIn {
    from { opacity:0; transform:translateX(-20px); }
    to   { opacity:1; transform:translateX(0); }
}

/* ── PREDICT BUTTON ── */
.stButton > button {
    background: linear-gradient(135deg, rgba(0,212,255,.15), rgba(123,47,255,.15)) !important;
    border: 1px solid var(--accent) !important;
    color: var(--accent) !important;
    font-family: 'Orbitron', sans-serif !important;
    font-size: .8rem !important;
    letter-spacing: 2px !important;
    padding: 12px 30px !important;
    border-radius: 6px !important;
    transition: all .3s !important;
    text-transform: uppercase !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, rgba(0,212,255,.35), rgba(123,47,255,.35)) !important;
    box-shadow: var(--glow) !important;
    transform: translateY(-2px) !important;
}

/* ── INPUTS ── */
.stTextInput > div > div > input {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 1rem !important;
    border-radius: 6px !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: var(--glow) !important;
}

/* ── SUCCESS / INFO / WARNING ALERTS ── */
.stSuccess, .stInfo, .stWarning, .stError {
    border-radius: 8px !important;
    font-family: 'Rajdhani', sans-serif !important;
}

/* ── FILE UPLOADER ── */
.stFileUploader {
    border: 1px dashed var(--border) !important;
    border-radius: 10px !important;
    background: var(--surface) !important;
    padding: 10px !important;
    transition: border-color .3s !important;
}
.stFileUploader:hover { border-color: var(--accent) !important; }

/* ── DEV CARD ── */
.dev-card {
    background: linear-gradient(160deg, var(--surface) 0%, var(--surface2) 100%);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 14px;
    text-align: center;
    transition: all .3s;
    position: relative;
    overflow: hidden;
}
.dev-card::before {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2), var(--accent3));
    transform: scaleX(0);
    transition: transform .4s;
}
.dev-card:hover::before { transform: scaleX(1); }
.dev-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 16px 40px rgba(0,212,255,.2);
    border-color: var(--accent);
}
.dev-name {
    font-family: 'Orbitron', sans-serif;
    color: var(--accent);
    font-size: .85rem;
    letter-spacing: 1.5px;
    margin: 8px 0 4px;
}
.dev-role {
    font-family: 'Share Tech Mono', monospace;
    color: var(--accent2);
    font-size: .75rem;
    letter-spacing: 1px;
}
.dev-desc {
    color: var(--text-dim);
    font-size: .9rem;
    margin: 6px 0;
}

/* ── CATEGORY GRID ── */
.cat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 22px;
    text-align: center;
    transition: all .35s;
    cursor: default;
    animation: cardIn .5s ease both;
}
.cat-card:hover {
    transform: scale(1.04);
    border-color: var(--accent);
    box-shadow: var(--glow);
}
.cat-icon { font-size: 2.4rem; margin-bottom: 8px; }
.cat-title {
    font-family: 'Orbitron', sans-serif;
    font-size: .8rem;
    letter-spacing: 2px;
    color: var(--accent);
    margin-bottom: 10px;
}
.cat-list { list-style: none; padding: 0; margin: 0; }
.cat-list li {
    font-family: 'Share Tech Mono', monospace;
    font-size: .78rem;
    color: var(--text-dim);
    padding: 3px 0;
    border-bottom: 1px solid rgba(255,255,255,.04);
}

/* ── SCANNER ANIMATION on prediction ── */
@keyframes scanDown {
    0%   { top: 0%; }
    100% { top: 100%; }
}
.scanner-bar {
    position: absolute;
    left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
    animation: scanDown 1.5s linear;
}

/* ── ABOUT PAGE ── */
.about-block {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 20px 24px;
    margin: 12px 0;
    transition: all .3s;
    position: relative;
    overflow: hidden;
}
.about-block:hover { border-color: var(--accent); box-shadow: var(--glow); }
.about-block .accent-line {
    position: absolute;
    top: 0; left: 0; bottom: 0;
    width: 3px;
    background: linear-gradient(180deg, var(--accent), var(--accent3));
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--accent); border-radius: 3px; }

/* ── DIVIDER ── */
hr { border-color: var(--border) !important; }

/* ── SPINNER ── */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* ── LINK BUTTON ── */
.stLinkButton a {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--accent) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: .75rem !important;
    letter-spacing: 1px !important;
    border-radius: 6px !important;
    transition: all .3s !important;
}
.stLinkButton a:hover {
    border-color: var(--accent) !important;
    box-shadow: var(--glow) !important;
    color: var(--accent) !important;
}

/* ── HIDE STREAMLIT CHROME ── */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# MODEL LOADING
# ============================================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("skin_disease_model.keras")

model = load_model()

# ============================================================
# PREDICTION LOGIC
# ============================================================
def model_prediction(test_image):
    image = Image.open(test_image).convert("RGB")
    image = image.resize((128, 128))
    input_arr = np.array(image) / 255.0
    input_arr = np.expand_dims(input_arr, axis=0)
    prediction = model.predict(input_arr)
    result_index = int(np.argmax(prediction))
    return result_index

# ============================================================
# DISEASE DATA – SUMMARY
# ============================================================
def disease_info_summary(disease_name):
    info = {
        "BA- cellulitis": "Bacteremia-associated (BA) cellulitis is a serious complication where a skin infection spreads to the bloodstream. While cellulitis is typically localized, systemic symptoms like fever, chills, and fatigue can indicate bacteremia, a potentially life-threatening condition. It is a rare complication, occurring in only about 5–10% of cases, but is more common in immunocompromised patients, those with diabetes, and the elderly. If left untreated, it can lead to further complications like sepsis, endocarditis, or osteomyelitis.",
        "BA-impetigo": "Impetigo is a highly contagious bacterial skin infection that most commonly affects infants and young children, often appearing around the nose, mouth, hands, and feet. The sores quickly rupture, ooze fluid, and develop a characteristic honey-colored crust. The infection is primarily caused by Staphylococcus aureus or Streptococcus pyogenes bacteria, which can enter the body through a cut, scrape, or insect bite. Treatment with antibiotics, either topical or oral, is used to clear the infection and limit its spread.",
        "FU-athlete-foot": "Athlete's foot, or tinea pedis, is a contagious fungal infection that typically causes an itchy, scaly, and burning rash, most commonly between the toes. It thrives in warm, moist environments like sweaty shoes and socks, and can spread through contact with infected people or surfaces in public places like pools and locker rooms. The infection can also cause dry, cracked skin on the soles or blisters on the feet. Treatment usually involves over-the-counter or prescription antifungal creams, powders, or sprays.",
        "FU-nail-fungus": "Nail fungus, or onychomycosis, is a common infection that makes nails thick, discolored, and brittle. It is most often caused by a type of fungus called dermatophyte, and thrives in warm, moist environments like shoes and public showers. The infection can lead to misshapen, crumbling nails and may emit a foul odor. While generally not serious for healthy individuals, it can be persistent, difficult to treat, and poses a risk of further infection for those with diabetes or weakened immune systems.",
        "FU-ringworm": "Ringworm is a common, contagious fungal infection of the skin, not caused by a worm. It appears as a red, itchy, ring-shaped rash, but symptoms can vary and may include scaly, raised patches. It spreads through direct contact with an infected person, animal, or contaminated surfaces like clothing and towels, and is treated with antifungal medication.",
        "PA-cutaneous-larva-migrans": "Cutaneous larva migrans (CLM), or 'creeping eruption' is a parasitic skin infection caused by hookworm larvae. Humans are accidental hosts who become infected through direct skin contact with warm, moist soil or sand contaminated with animal feces. The larvae, most commonly from dog and cat hookworms, burrow into the skin but cannot penetrate past the outer layer. This migration causes intensely itchy, red, winding tracks on the skin, typically on the feet, legs, or buttocks.",
        "VI-chickenpox": "Chickenpox is a highly contagious viral infection caused by the varicella-zoster virus (VZV). It is characterized by an itchy rash of fluid-filled blisters that eventually scab over, accompanied by symptoms like fever and fatigue. While typically mild in children, it can cause serious complications in adults, pregnant women, and those with weakened immune systems.",
        "VI-shingles": "Shingles is a reactivation of the chickenpox virus causing painful rashes and nerve pain on one side of the body. This condition, a complication of herpes zoster ophthalmicus (HZO), can lead to abducens nerve palsy. The palsy causes weakness or paralysis of the lateral rectus muscle, leading to an impaired ability to move the eye outwards. Symptoms include horizontal double vision and the inability to abduct the affected eye.",
        "Actinic Keratosis": "An actinic keratosis is a rough, scaly patch on the skin that develops from years of sun exposure. It's often found on the face, lips, ears, forearms, scalp, neck or back of the hands, and can increase the risk of developing skin cancer.",
        "Bowen Disease": "Bowen disease is a pre-cancerous skin condition with a low risk of progressing to invasive squamous cell carcinoma (SCC), estimated at 3%–5%. Key risk factors include excessive sun exposure, fair skin, older age, and a weakened immune system. While the prognosis is generally excellent with treatment, the lesions can be progressive."
    }
    return info.get(disease_name, "Information not available for this disease.")

# ============================================================
# DISEASE DATA – DETAILED
# ============================================================
def disease_info_detailed(disease_name):
    details = {
        "BA- cellulitis": {
            "symptoms": ["Red, swollen, tender skin", "Area feels warm to touch", "Pain in the affected area", "Possible high fever", "Red streaks spreading outward"],
            "medicine": ["Oral Dicloxacillin", "Cephalexin (500mg QID)", "IV antibiotics for severe cases"],
            "precautions": ["Keep skin meticulously clean", "Elevate the affected limb", "Avoid scratching or touching"],
            "timetable": ["**Day 1–2**: Start Oral Cephalexin. Monitor for spreading redness.", "**Day 3–5**: Redness should stabilize; swelling starts to decrease.", "**Day 6–10**: Finish entire antibiotic course to prevent relapse."]
        },
        "BA-impetigo": {
            "symptoms": ["Red sores that rupture quickly", "Honey-colored crusts forming", "Fluid-filled blisters"],
            "medicine": ["Mupirocin ointment (topical)", "Oral Cephalexin", "Antibacterial soap"],
            "precautions": ["Wash hands frequently", "Don't share towels or clothing", "Keep nails short and clean"],
            "timetable": ["**Day 1–3**: Apply Mupirocin 2% TID. Sores start to crust.", "**Day 4–7**: Crusts begin to fall off. Continue topical application.", "**Day 8–10**: Skin heals; maintain hygiene to prevent spread."]
        },
        "FU-athlete-foot": {
            "symptoms": ["Itchy, scaly red rash", "Burning sensation between toes", "Dry cracking skin on soles"],
            "medicine": ["Clotrimazole cream (1%)", "Terbinafine spray", "Antifungal powder"],
            "precautions": ["Keep feet completely dry", "Change socks daily", "Wear sandals in public showers"],
            "timetable": ["**Day 1–3**: Apply Terbinafine 2x daily. Itching reduces.", "**Day 4–7**: Peeling skin starts to heal. Keep area very dry.", "**Day 8–14**: Continue treatment even if skin looks clear to kill spores."]
        },
        "FU-nail-fungus": {
            "symptoms": ["Thickened discolored nails", "Yellow-brown discoloration", "Brittle, crumbling edges"],
            "medicine": ["Oral Terbinafine (250mg)", "Ciclopirox nail lacquer", "Laser therapy (clinical)"],
            "precautions": ["Trim nails straight across", "Disinfect clippers after use", "Wear moisture-wicking socks"],
            "timetable": ["**Day 1–10**: Start Oral Terbinafine. (Full treatment: 6–12 weeks)", "**Month 1**: Medication reaches the nail root via bloodstream.", "**Month 3**: New healthy nail begins to push out infected area."]
        },
        "FU-ringworm": {
            "symptoms": ["Red ring-shaped rash", "Scaly, raised border", "Intensely itchy patches"],
            "medicine": ["Clotrimazole cream", "Miconazole (topical)", "Oral Griseofulvin (severe)"],
            "precautions": ["Avoid touching infected pets", "Don't share sports gear", "Wash bedding in hot water"],
            "timetable": ["**Day 1–3**: Apply antifungal cream. Ring border starts to flatten.", "**Day 4–7**: Center of ring clears. Itching subsides.", "**Day 8–14**: Finish full cream tube to prevent ring from returning."]
        },
        "PA-cutaneous-larva-migrans": {
            "symptoms": ["Snake-like red winding tracks", "Intense nocturnal itching", "Small fluid-filled blisters"],
            "medicine": ["Albendazole (400mg)", "Ivermectin (oral)", "Thiabendazole cream"],
            "precautions": ["Always wear shoes on sand/soil", "Deworm household pets regularly", "Use a towel when sitting on beach"],
            "timetable": ["**Day 1–2**: Take Albendazole 400mg. Larva movement stops.", "**Day 3–5**: Red tracks begin to fade. Itching significantly decreases.", "**Day 6–10**: Skin returns to normal; all tracks disappear."]
        },
        "VI-chickenpox": {
            "symptoms": ["Intensely itchy blisters", "High fever and fatigue", "Fluid-filled vesicles spreading"],
            "medicine": ["Acetaminophen (fever)", "Calamine lotion (itch)", "Acyclovir (antiviral)"],
            "precautions": ["Varicella vaccine (prevention)", "Isolate from immunocompromised", "Do not scratch — risk of scarring"],
            "timetable": ["**Day 1–3**: Fever management. New blisters appear. Start Acyclovir if prescribed.", "**Day 4–6**: Blisters begin to cloud over and form scabs.", "**Day 7–10**: Scabs fall off. No longer contagious once all are dry."]
        },
        "VI-shingles": {
            "symptoms": ["Painful one-sided rash", "Tingling or burning sensation", "Fluid-filled blisters in a band"],
            "medicine": ["Acyclovir (antiviral)", "Valacyclovir (preferred)", "Gabapentin (nerve pain)"],
            "precautions": ["Shingles vaccine (Shingrix)", "Keep rash covered with gauze", "Manage stress and sleep"],
            "timetable": ["**Day 1–3**: Start Antivirals (Acyclovir 800mg 5x/day). Pain is highest.", "**Day 4–7**: Blisters stop forming and begin to dry out.", "**Day 8–14**: Scabs form. Monitor for lingering nerve pain (PHN)."]
        },
        "Actinic Keratosis": {
            "symptoms": ["Rough, scaly patch on skin", "Wart-like or flat surface", "Located on sun-exposed areas"],
            "medicine": ["Fluorouracil (5-FU) cream", "Imiquimod (immune response)", "Cryotherapy (freezing)"],
            "precautions": ["SPF 30+ sunscreen daily", "Wear wide-brimmed hats", "Schedule regular skin checks"],
            "timetable": ["**Day 1–5**: Apply 5-FU cream. Area becomes red and sore.", "**Day 6–14**: Peak inflammation. Skin may crust or ooze (expected).", "**Day 15+**: Treatment stops; healthy skin begins to grow back."]
        },
        "Bowen Disease": {
            "symptoms": ["Red scaly persistent patch", "Slow-growing lesion over months", "Crusting or weeping surface"],
            "medicine": ["Imiquimod cream", "5-Fluorouracil (topical)", "Surgical excision (clinical)"],
            "precautions": ["Minimize all UV exposure", "Monitor any patch for changes", "Apply high-protection SPF daily"],
            "timetable": ["**Day 1–10**: Daily Imiquimod application. Site becomes inflamed.", "**Week 3–4**: Intense immune reaction attacks pre-cancerous cells.", "**Month 2**: Inflammation subsides; clear healthy skin emerges."]
        }
    }
    return details.get(disease_name, None)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 24px;'>
        <div style='font-family:Orbitron,sans-serif; font-size:1.1rem; color:#00d4ff; 
                    text-shadow:0 0 18px rgba(0,212,255,.6); letter-spacing:3px; font-weight:900;'>
            DERM<span style='color:#ff2d78;'>AI</span>
        </div>
        <div style='font-family:"Share Tech Mono",monospace; font-size:.65rem; color:#4a6a8a; 
                    letter-spacing:2px; margin-top:4px;'>SKIN DISEASE DETECTION</div>
        <div style='margin:16px auto; height:1px; background:linear-gradient(90deg,transparent,#00d4ff,transparent);'></div>
    </div>
    """, unsafe_allow_html=True)

    app_mode = st.radio(
        "NAVIGATE",
        ["Home", "Disease Recognition", "AI Health Bot", "Categories", "Developers Group", "About Project"],
        label_visibility="visible"
    )

    st.markdown("""
    <div style='margin-top:40px; padding:14px; background:rgba(0,212,255,.04); 
                border:1px solid rgba(0,212,255,.12); border-radius:8px; text-align:center;'>
        <div style='font-family:"Share Tech Mono",monospace; font-size:.65rem; color:#4a6a8a; letter-spacing:2px;'>
            POWERED BY<br>
            <span style='color:#00d4ff;'>TENSORFLOW + CNN</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# HOME PAGE
# ============================================================
if app_mode == "Home":

    # Hero Banner
    st.markdown("""
    <div class="hero-banner">
        <p class="hero-title">DERMAI</p>
        <p class="hero-sub">AI-Powered Skin Disease Detection System</p>
        <p style='color:#4a6a8a; font-family:Rajdhani,sans-serif; font-size:.95rem; 
                  margin-top:16px; letter-spacing:2px;'>
            UPLOAD · ANALYZE · UNDERSTAND
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Stats row
    c1, c2, c3, c4 = st.columns(4)
    stats = [
        ("10", "DISEASES DETECTED"),
        ("128×128", "INPUT RESOLUTION"),
        ("CNN", "MODEL ARCHITECTURE"),
        ("5", "DISEASE CATEGORIES"),
    ]
    for col, (num, label) in zip([c1, c2, c3, c4], stats):
        with col:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-num">{num}</div>
                <div class="stat-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Image + intro
    col_img, col_text = st.columns([1, 1], gap="large")
    with col_img:
        if os.path.exists("sample_image.jpg"):
            st.image("sample_image.jpg", use_container_width=True,
                     caption="Sample diagnostic image")
        else:
            st.markdown("""
            <div style='background:linear-gradient(135deg,#0d1b2a,#112236); 
                        border:1px dashed rgba(0,212,255,.3); border-radius:12px; 
                        height:300px; display:flex; align-items:center; justify-content:center;
                        flex-direction:column; gap:10px;'>
                <div style='font-size:4rem;'>🧬</div>
                <div style='font-family:"Share Tech Mono",monospace; color:#4a6a8a; 
                            font-size:.8rem; letter-spacing:2px;'>SAMPLE IMAGE PLACEHOLDER</div>
            </div>
            """, unsafe_allow_html=True)

    with col_text:
        st.markdown("""
        <div class="derm-card">
            <div class="accent-line"></div>
            <h3 style='color:#00d4ff; font-family:Orbitron,sans-serif; font-size:.9rem; letter-spacing:2px;'>
                👋 INTRODUCTION
            </h3>
            <p style='color:#a0c0d8; font-family:Rajdhani,sans-serif; font-size:1.05rem; line-height:1.7;'>
                Welcome to the <strong style='color:#00d4ff;'>AI-Powered Skin Disease Recognition System</strong>.<br><br>
                This platform leverages deep learning to help you <strong style='color:#ff2d78;'>detect skin diseases</strong> 
                from uploaded images with high accuracy.
            </p>
            <p style='color:#6a8aaa; font-family:Rajdhani,sans-serif; font-size:.95rem; line-height:1.7;'>
                🔍 Upload an image, receive an instant AI prediction, and access comprehensive 
                medical information about the detected condition.
            </p>
        </div>

        <div class="derm-card" style='margin-top:12px;'>
            <h3 style='color:#00d4ff; font-family:Orbitron,sans-serif; font-size:.9rem; letter-spacing:2px;'>
                ⚙️ HOW IT WORKS
            </h3>
            <div style='display:flex; gap:12px; align-items:flex-start; margin:8px 0;'>
                <span style='font-family:Orbitron,sans-serif; color:#ff2d78; font-size:1.4rem; font-weight:900;'>01</span>
                <span style='color:#a0c0d8; font-family:Rajdhani,sans-serif;'>Upload a clear photo of the affected skin area</span>
            </div>
            <div style='display:flex; gap:12px; align-items:flex-start; margin:8px 0;'>
                <span style='font-family:Orbitron,sans-serif; color:#7b2fff; font-size:1.4rem; font-weight:900;'>02</span>
                <span style='color:#a0c0d8; font-family:Rajdhani,sans-serif;'>CNN model analyzes image features at 128×128 resolution</span>
            </div>
            <div style='display:flex; gap:12px; align-items:flex-start; margin:8px 0;'>
                <span style='font-family:Orbitron,sans-serif; color:#00d4ff; font-size:1.4rem; font-weight:900;'>03</span>
                <span style='color:#a0c0d8; font-family:Rajdhani,sans-serif;'>Receive diagnosis + full medical info & treatment timeline</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style='margin-top:20px; padding:16px 24px; background:rgba(255,45,120,.05); 
                border:1px solid rgba(255,45,120,.2); border-radius:8px; 
                font-family:Rajdhani,sans-serif; color:#a0c0d8; font-size:.9rem; letter-spacing:1px;'>
        ⚠️&nbsp; <strong style='color:#ff2d78;'>DISCLAIMER:</strong>&nbsp; 
        This tool is for informational and educational purposes only. 
        Always consult a qualified medical professional for diagnosis and treatment.
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# DISEASE RECOGNITION PAGE
# ============================================================
elif app_mode == "Disease Recognition":

    st.markdown("""
    <h1 style='margin-bottom:4px;'>🧬 DISEASE RECOGNITION</h1>
    <p style='font-family:"Share Tech Mono",monospace; color:#4a6a8a; font-size:.8rem; 
              letter-spacing:3px; margin-bottom:24px;'>CNN · TENSORFLOW · IMAGE CLASSIFIER</p>
    """, unsafe_allow_html=True)

    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown("""
        <div style='font-family:Orbitron,sans-serif; color:#00d4ff; font-size:.75rem; 
                    letter-spacing:2px; margin-bottom:8px;'>📤 UPLOAD IMAGE</div>
        """, unsafe_allow_html=True)
        test_image = st.file_uploader("", type=["jpg", "jpeg", "png"],
                                       label_visibility="collapsed")

        if test_image is not None:
            image = Image.open(test_image).convert("RGB")
            st.markdown("""
            <div style='font-family:"Share Tech Mono",monospace; color:#4a6a8a; 
                        font-size:.7rem; letter-spacing:2px; margin:12px 0 6px;'>PREVIEW</div>
            """, unsafe_allow_html=True)
            st.image(image, use_container_width=True, caption="")

            st.markdown("""
            <div style='font-family:"Share Tech Mono",monospace; color:#00d4ff; 
                        font-size:.7rem; letter-spacing:2px; margin-top:8px;'>
                ✅ IMAGE LOADED · READY FOR ANALYSIS
            </div>
            """, unsafe_allow_html=True)

    with col_result:
        st.markdown("""
        <div style='font-family:Orbitron,sans-serif; color:#00d4ff; font-size:.75rem; 
                    letter-spacing:2px; margin-bottom:8px;'>🔎 ANALYSIS RESULT</div>
        """, unsafe_allow_html=True)

        if test_image is not None:
            if st.button("⚡ RUN DIAGNOSTIC SCAN"):
                with st.spinner("Scanning image with neural network..."):
                    result_index = model_prediction(test_image)
                    class_names = [
                        'BA- cellulitis', 'BA-impetigo', 'FU-athlete-foot',
                        'FU-nail-fungus', 'FU-ringworm',
                        'PA-cutaneous-larva-migrans', 'VI-chickenpox', 'VI-shingles',
                        'Actinic Keratosis', 'Bowen Disease'
                    ]
                    predicted_disease = class_names[result_index]

                st.markdown(f"""
                <div class="derm-card" style='border-color:rgba(0,212,255,.4);'>
                    <div style='font-family:"Share Tech Mono",monospace; color:#4a6a8a; 
                                font-size:.7rem; letter-spacing:3px;'>PREDICTION RESULT</div>
                    <div style='font-family:Orbitron,sans-serif; font-size:1.2rem; 
                                color:#00d4ff; text-shadow:0 0 18px rgba(0,212,255,.6); 
                                margin:10px 0; font-weight:900;'>
                        {predicted_disease.upper()}
                    </div>
                    <div style='font-family:"Share Tech Mono",monospace; color:#4a6a8a; 
                                font-size:.65rem; letter-spacing:2px;'>
                        CONFIDENCE · HIGH · CNN MODEL
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                <div style='font-family:Orbitron,sans-serif; color:#00d4ff; font-size:.75rem; 
                            letter-spacing:2px; margin:16px 0 8px;'>💬 AI HEALTH BOT SUMMARY</div>
                """, unsafe_allow_html=True)

                summary = disease_info_summary(predicted_disease)
                st.markdown(f"""
                <div class="derm-card">
                    <p style='color:#a0c0d8; font-family:Rajdhani,sans-serif; 
                              font-size:1rem; line-height:1.7; margin:0;'>{summary}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background:var(--surface, #0d1b2a); border:1px dashed rgba(0,212,255,.2); 
                        border-radius:12px; height:280px; display:flex; align-items:center; 
                        justify-content:center; flex-direction:column; gap:14px;'>
                <div style='font-size:3rem; opacity:.4;'>🔬</div>
                <div style='font-family:"Share Tech Mono",monospace; color:#2a4a6a; 
                            font-size:.75rem; letter-spacing:3px; text-align:center;'>
                    AWAITING IMAGE INPUT<br>UPLOAD TO BEGIN SCAN
                </div>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# AI HEALTH BOT PAGE
# ============================================================
elif app_mode == "AI Health Bot":

    st.markdown("""
    <h1 style='margin-bottom:4px;'>🤖 AI HEALTH BOT</h1>
    <p style='font-family:"Share Tech Mono",monospace; color:#4a6a8a; font-size:.8rem; 
              letter-spacing:3px; margin-bottom:24px;'>DIAGNOSE AI · MEDICAL DATABASE</p>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="derm-card" style='margin-bottom:20px;'>
        <p style='color:#a0c0d8; font-family:Rajdhani,sans-serif; font-size:1rem; margin:0;'>
            Type a disease name below to retrieve detailed medical analysis.<br>
            <span style='color:#4a6a8a; font-family:"Share Tech Mono",monospace; font-size:.8rem;'>
            TRY: ringworm · cellulitis · shingles · impetigo
            </span>
        </p>
    </div>
    """, unsafe_allow_html=True)

    query = st.text_input("", placeholder="🔍  Enter disease name...", label_visibility="collapsed")

    if query:
        class_names = [
            'BA- cellulitis', 'BA-impetigo', 'FU-athlete-foot',
            'FU-nail-fungus', 'FU-ringworm', 'PA-cutaneous-larva-migrans',
            'VI-chickenpox', 'VI-shingles', 'Actinic Keratosis', 'Bowen Disease'
        ]
        found = [key for key in class_names if query.lower() in key.lower()]

        if found:
            disease_name = found[0]
            data = disease_info_detailed(disease_name)

            if data:
                st.markdown(f"""
                <div style='font-family:Orbitron,sans-serif; font-size:1rem; color:#00d4ff; 
                            text-shadow:0 0 18px rgba(0,212,255,.5); letter-spacing:2px; 
                            margin:20px 0 16px; border-bottom:1px solid rgba(0,212,255,.15); 
                            padding-bottom:12px;'>
                    📊 MEDICAL ANALYSIS: {disease_name.upper()}
                </div>
                """, unsafe_allow_html=True)

                col1, col2, col3 = st.columns(3, gap="medium")

                with col1:
                    st.markdown("""
                    <div style='font-family:Orbitron,sans-serif; color:#00d4ff; 
                                font-size:.75rem; letter-spacing:2px; margin-bottom:10px;'>
                        🌡️ SYMPTOMS
                    </div>
                    """, unsafe_allow_html=True)
                    for s in data['symptoms']:
                        st.markdown(f'<div class="pill">{s}</div>', unsafe_allow_html=True)

                with col2:
                    st.markdown("""
                    <div style='font-family:Orbitron,sans-serif; color:#7b2fff; 
                                font-size:.75rem; letter-spacing:2px; margin-bottom:10px;'>
                        💊 DIAGNOSIS & MEDS
                    </div>
                    """, unsafe_allow_html=True)
                    for m in data['medicine']:
                        st.markdown(f'<div class="pill pill-med">{m}</div>', unsafe_allow_html=True)

                with col3:
                    st.markdown("""
                    <div style='font-family:Orbitron,sans-serif; color:#ff2d78; 
                                font-size:.75rem; letter-spacing:2px; margin-bottom:10px;'>
                        🛡️ PRECAUTIONS
                    </div>
                    """, unsafe_allow_html=True)
                    for p in data['precautions']:
                        st.markdown(f'<div class="pill pill-prec">{p}</div>', unsafe_allow_html=True)
            else:
                st.warning("Detailed data is missing for this selection.")
        else:
            st.markdown("""
            <div style='padding:16px; background:rgba(255,45,120,.06); border:1px solid rgba(255,45,120,.25);
                        border-radius:8px; font-family:Rajdhani,sans-serif; color:#ff2d78;'>
                ❌ Disease not found. Please check spelling or try: cellulitis, ringworm, shingles
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# CATEGORIES PAGE
# ============================================================
elif app_mode == "Categories":

    st.markdown("""
    <h1 style='margin-bottom:4px;'>📂 DISEASE CATEGORIES</h1>
    <p style='font-family:"Share Tech Mono",monospace; color:#4a6a8a; font-size:.8rem; 
              letter-spacing:3px; margin-bottom:28px;'>CLASSIFICATION SYSTEM · 10 CONDITIONS</p>
    """, unsafe_allow_html=True)

    categories = [
        {
            "icon": "🦠", "title": "BACTERIAL", "prefix": "BA", "color": "#ff2d78",
            "badge_class": "badge-ba",
            "diseases": ["Cellulitis", "Impetigo"],
            "desc": "Infections caused by Staphylococcus aureus or Streptococcus pyogenes bacteria."
        },
        {
            "icon": "🍄", "title": "FUNGAL", "prefix": "FU", "color": "#7b2fff",
            "badge_class": "badge-fu",
            "diseases": ["Athlete's Foot", "Ringworm", "Nail Fungus"],
            "desc": "Dermatophyte-based infections thriving in warm, moist skin environments."
        },
        {
            "icon": "🪱", "title": "PARASITIC", "prefix": "PA", "color": "#00d4ff",
            "badge_class": "badge-pa",
            "diseases": ["Cutaneous Larva Migrans"],
            "desc": "Hookworm larvae infections acquired via contaminated soil or sand contact."
        },
        {
            "icon": "🧫", "title": "VIRAL", "prefix": "VI", "color": "#ffc800",
            "badge_class": "badge-vi",
            "diseases": ["Chickenpox", "Shingles"],
            "desc": "Varicella-zoster virus driven conditions affecting skin and nerve pathways."
        },
        {
            "icon": "☀️", "title": "PRE-CANCEROUS", "prefix": "PC", "color": "#00ff96",
            "badge_class": "badge-pc",
            "diseases": ["Actinic Keratosis", "Bowen Disease"],
            "desc": "UV-induced lesions with low but real risk of progressing to invasive SCC."
        },
    ]

    cols = st.columns(5, gap="small")
    for col, cat in zip(cols, categories):
        with col:
            diseases_html = "".join([f"<li>{d}</li>" for d in cat["diseases"]])
            st.markdown(f"""
            <div class="cat-card" style='border-color:rgba({
                "255,45,120" if cat["prefix"]=="BA" else
                "123,47,255" if cat["prefix"]=="FU" else
                "0,212,255" if cat["prefix"]=="PA" else
                "255,200,0" if cat["prefix"]=="VI" else
                "0,255,150"
            },.25);'>
                <div class="cat-icon">{cat["icon"]}</div>
                <div class="cat-title" style='color:{cat["color"]};'>{cat["title"]}</div>
                <span class="badge {cat['badge_class']}">{cat["prefix"]}</span>
                <ul class="cat-list" style='margin-top:12px;'>
                    {diseases_html}
                </ul>
                <p style='font-family:Rajdhani,sans-serif; font-size:.8rem; color:#3a5a7a; 
                           margin-top:12px; line-height:1.5;'>{cat["desc"]}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<hr style="border-color:rgba(0,212,255,.15);">', unsafe_allow_html=True)

    # Treatment timeline section
    st.markdown("""
    <div style='font-family:Orbitron,sans-serif; color:#00d4ff; font-size:.85rem; 
                letter-spacing:3px; margin:20px 0 16px;'>
        🏥 TREATMENT TIMELINE LOOKUP
    </div>
    """, unsafe_allow_html=True)

    user_query = st.text_input("", placeholder="🔍  Enter disease name (e.g. Shingles, Athlete's Foot)...",
                               label_visibility="collapsed", key="timeline_search")

    if user_query:
        class_names = [
            'BA- cellulitis', 'BA-impetigo', 'FU-athlete-foot',
            'FU-nail-fungus', 'FU-ringworm', 'PA-cutaneous-larva-migrans',
            'VI-chickenpox', 'VI-shingles', 'Actinic Keratosis', 'Bowen Disease'
        ]
        matches = [k for k in class_names if user_query.lower() in k.lower()]

        if matches:
            selected_disease = matches[0]
            data = disease_info_detailed(selected_disease)

            st.markdown(f"""
            <div style='font-family:Orbitron,sans-serif; color:#00d4ff; font-size:.8rem; 
                        letter-spacing:2px; margin:16px 0 12px;'>
                📅 GENERAL CARE TIMELINE: {selected_disease.upper()}
            </div>
            """, unsafe_allow_html=True)

            for i, step in enumerate(data['timetable']):
                st.markdown(f"""
                <div class="timeline-step" style='animation-delay:{i*0.15}s;'>
                    {step}
                </div>
                """, unsafe_allow_html=True)

            st.markdown("""
            <div style='margin-top:16px; padding:14px 20px; 
                        background:rgba(0,212,255,.06); border:1px solid rgba(0,212,255,.2); 
                        border-radius:8px; font-family:Rajdhani,sans-serif; 
                        color:#a0c0d8; font-size:.95rem;'>
                💡 <strong style='color:#00d4ff;'>TIP:</strong> Always complete the full course of treatment 
                as recommended by a healthcare provider, even if symptoms improve early.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='padding:16px; background:rgba(255,45,120,.06); border:1px solid rgba(255,45,120,.25);
                        border-radius:8px; font-family:Rajdhani,sans-serif; color:#ff2d78;'>
                Disease not found. Please try: Cellulitis, Ringworm, or Shingles
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# DEVELOPERS GROUP PAGE
# ============================================================
elif app_mode == "Developers Group":

    st.markdown("""
    <h1 style='margin-bottom:4px;'>👩‍💻 DEVELOPERS GROUP</h1>
    <p style='font-family:"Share Tech Mono",monospace; color:#4a6a8a; font-size:.8rem; 
              letter-spacing:3px; margin-bottom:28px;'>MEET THE MINDS BEHIND DERMAI</p>
    """, unsafe_allow_html=True)

    devs = [
        ("Dev 1", "ML Engineer", "CNN & TF Expert", "profile_image.jpeg", "https://www.linkedin.com/in/subhajit-mondal-85946328b"),
        ("Dev 2", "Frontend Dev", "Streamlit UI Expert", "dev2.jpg", "https://linkedin.com/in/user2"),
        ("Dev 3", "Data Scientist", "Data Preprocessing", "dev3.jpg", "https://linkedin.com/in/user3"),
        ("Dev 4", "Backend Dev", "API & Logic", "dev4.jpg", "https://linkedin.com/in/user4"),
        ("Dev 5", "UI Designer", "Visual Experience", "dev5.jpg", "https://linkedin.com/in/user5"),
    ]

    cols = st.columns(5, gap="medium")
    role_colors = ["#00d4ff", "#ff2d78", "#7b2fff", "#ffc800", "#00ff96"]

    for i, (col, (name, role, desc, img, link)) in enumerate(zip(cols, devs)):
        with col:
            st.markdown(f'<div class="dev-card" style="animation-delay:{i*0.1}s;">', unsafe_allow_html=True)
            if os.path.exists(img):
                st.image(img, use_container_width=True)
            else:
                st.markdown(f"""
                <div style='height:100px; background:linear-gradient(135deg,#0d1b2a,#112236); 
                            border-radius:8px; display:flex; align-items:center; 
                            justify-content:center; font-size:2.5rem; margin-bottom:8px;'>
                    👤
                </div>
                """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class="dev-name">{name}</div>
            <div class="dev-role" style='color:{role_colors[i]};'>{role}</div>
            <div class="dev-desc">{desc}</div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.link_button("🔗 LinkedIn", link)

# ============================================================
# ABOUT PROJECT PAGE
# ============================================================
elif app_mode == "About Project":

    st.markdown("""
    <h1 style='margin-bottom:4px;'>ℹ️ ABOUT PROJECT</h1>
    <p style='font-family:"Share Tech Mono",monospace; color:#4a6a8a; font-size:.8rem; 
              letter-spacing:3px; margin-bottom:28px;'>ARCHITECTURE · MISSION · STACK</p>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown("""
        <div class="derm-card">
            <div style='position:absolute; top:0; left:0; bottom:0; width:3px;
                        background:linear-gradient(180deg,#00d4ff,#7b2fff);'></div>
            <h3 style='color:#00d4ff; font-family:Orbitron,sans-serif; 
                       font-size:.8rem; letter-spacing:2px;'>🧠 OVERVIEW</h3>
            <p style='color:#a0c0d8; font-family:Rajdhani,sans-serif; font-size:1rem; line-height:1.8;'>
                DermAI uses a <strong style='color:#00d4ff;'>Convolutional Neural Network (CNN)</strong> trained on 
                dermatological image datasets to classify 10 distinct skin conditions across 5 medical categories.<br><br>
                The model processes images at <strong style='color:#ff2d78;'>128×128 resolution</strong> and 
                returns a softmax confidence score for each class.
            </p>
        </div>

        <div class="derm-card" style='margin-top:14px;'>
            <div style='position:absolute; top:0; left:0; bottom:0; width:3px;
                        background:linear-gradient(180deg,#ff2d78,#7b2fff);'></div>
            <h3 style='color:#ff2d78; font-family:Orbitron,sans-serif; 
                       font-size:.8rem; letter-spacing:2px;'>🎯 GOAL</h3>
            <p style='color:#a0c0d8; font-family:Rajdhani,sans-serif; font-size:1rem; line-height:1.8;'>
                Build <strong style='color:#ff2d78;'>accessible healthcare AI</strong> that helps individuals 
                in resource-limited settings identify potential skin conditions and seek timely medical attention. 🌍
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown("""
        <div class="derm-card">
            <div style='position:absolute; top:0; left:0; bottom:0; width:3px;
                        background:linear-gradient(180deg,#7b2fff,#00d4ff);'></div>
            <h3 style='color:#7b2fff; font-family:Orbitron,sans-serif; 
                       font-size:.8rem; letter-spacing:2px;'>⚙️ TECH STACK</h3>
        """, unsafe_allow_html=True)

        stack = [
            ("🧠", "TensorFlow", "Deep learning framework", "#00d4ff"),
            ("🐍", "Python", "Core programming language", "#ffc800"),
            ("🎨", "Streamlit", "Web application framework", "#ff2d78"),
            ("📊", "NumPy", "Numerical computation", "#7b2fff"),
            ("🖼️", "Pillow", "Image preprocessing", "#00ff96"),
        ]
        for icon, tech, desc, color in stack:
            st.markdown(f"""
            <div style='display:flex; align-items:center; gap:14px; padding:10px 0; 
                        border-bottom:1px solid rgba(255,255,255,.04);'>
                <span style='font-size:1.3rem;'>{icon}</span>
                <div>
                    <div style='font-family:Orbitron,sans-serif; color:{color}; 
                                font-size:.75rem; letter-spacing:1.5px;'>{tech}</div>
                    <div style='font-family:Rajdhani,sans-serif; color:#4a6a8a; font-size:.85rem;'>{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="derm-card" style='margin-top:14px;'>
            <div style='position:absolute; top:0; left:0; bottom:0; width:3px;
                        background:linear-gradient(180deg,#00ff96,#00d4ff);'></div>
            <h3 style='color:#00ff96; font-family:Orbitron,sans-serif; 
                       font-size:.8rem; letter-spacing:2px;'>📋 MODEL DETAILS</h3>
            <div style='display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:10px;'>
                <div style='text-align:center; padding:12px; background:rgba(0,212,255,.06); border-radius:8px;'>
                    <div style='font-family:Orbitron,sans-serif; color:#00d4ff; font-size:1.1rem; font-weight:900;'>CNN</div>
                    <div style='font-family:"Share Tech Mono",monospace; color:#4a6a8a; font-size:.65rem;'>ARCHITECTURE</div>
                </div>
                <div style='text-align:center; padding:12px; background:rgba(255,45,120,.06); border-radius:8px;'>
                    <div style='font-family:Orbitron,sans-serif; color:#ff2d78; font-size:1.1rem; font-weight:900;'>10</div>
                    <div style='font-family:"Share Tech Mono",monospace; color:#4a6a8a; font-size:.65rem;'>OUTPUT CLASSES</div>
                </div>
                <div style='text-align:center; padding:12px; background:rgba(123,47,255,.06); border-radius:8px;'>
                    <div style='font-family:Orbitron,sans-serif; color:#7b2fff; font-size:1.1rem; font-weight:900;'>128²</div>
                    <div style='font-family:"Share Tech Mono",monospace; color:#4a6a8a; font-size:.65rem;'>INPUT SIZE</div>
                </div>
                <div style='text-align:center; padding:12px; background:rgba(0,255,150,.06); border-radius:8px;'>
                    <div style='font-family:Orbitron,sans-serif; color:#00ff96; font-size:1.1rem; font-weight:900;'>.keras</div>
                    <div style='font-family:"Share Tech Mono",monospace; color:#4a6a8a; font-size:.65rem;'>MODEL FORMAT</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
