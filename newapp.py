import os
import gdown
import tensorflow as tf
import streamlit as st

# --- Download model if it doesn't exist locally ---
model_path = "skin_disease_model (1).keras"
if not os.path.exists(model_path):
    url = "https://drive.google.com/uc?id=167eTthYXh3ogLTf3s5S6IHhJMF-ys6a0"
    gdown.download(url, model_path, quiet=False)

# --- Load model with caching ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(model_path)

model = load_model()

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("skin_disease_model.keras")

model = load_model()

# --- PREDICTION LOGIC ---
def model_prediction(test_image):
    image = Image.open(test_image).convert("RGB")
    image = image.resize((128, 128))
    input_arr = np.array(image) / 255.0
    input_arr = np.expand_dims(input_arr, axis=0)
    prediction = model.predict(input_arr)
    result_index = int(np.argmax(prediction))
    return result_index

# --- DATA FOR RECOGNITION (Original Paragraphs) ---
def disease_info_summary(disease_name):
    info = {
        "BA- cellulitis": "Bacteremia-associated (BA) cellulitis is a serious complication where a skin infection spreads to the bloodstream. While cellulitis is typically localized, systemic symptoms like fever, chills, and fatigue can indicate bacteremia, a potentially life-threatening condition. It is a rare complication, occurring in only about 5–10% of cases, but is more common in immunocompromised patients, those with diabetes, and the elderly. If left untreated, it can lead to further complications like sepsis, endocarditis, or osteomyelitis.",
        "BA-impetigo": "Impetigo is a highly contagious bacterial skin infection that most commonly affects infants and young children, often appearing around the nose, mouth, hands, and feet. The sores quickly rupture, ooze fluid, and develop a characteristic honey-colored crust. The infection is primarily caused by Staphylococcus aureus or Streptococcus pyogenes bacteria, which can enter the body through a cut, scrape, or insect bite. Treatment with antibiotics, either topical or oral, is used to clear the infection and limit its spread",
        "FU-athlete-foot": "Athlete's foot, or tinea pedis, is a contagious fungal infection that typically causes an itchy, scaly, and burning rash, most commonly between the toes. It thrives in warm, moist environments like sweaty shoes and socks, and can spread through contact with infected people or surfaces in public places like pools and locker rooms. The infection can also cause dry, cracked skin on the soles or blisters on the feet. Treatment usually involves over-the-counter or prescription antifungal creams, powders, or sprays.",
        "FU-nail-fungus": "Nail fungus, or onychomycosis, is a common infection that makes nails thick, discolored, and brittle. It is most often caused by a type of fungus called dermatophyte, and thrives in warm, moist environments like shoes and public showers. The infection can lead to misshapen, crumbling nails and may emit a foul odor. While generally not serious for healthy individuals, it can be persistent, difficult to treat, and poses a risk of further infection for those with diabetes or weakened immune systems.",
        "FU-ringworm": "Ringworm is a common, contagious fungal infection of the skin, not caused by a worm. It appears as a red, itchy, ring-shaped rash, but symptoms can vary and may include scaly, raised patches. It spreads through direct contact with an infected person, animal, or contaminated surfaces like clothing and towels, and is treated with antifungal medication.",
        "PA-cutaneous-larva-migrans": "Cutaneous larva migrans (CLM), or 'creeping eruption' is a parasitic skin infection caused by hookworm larvae. Humans are accidental hosts who become infected through direct skin contact with warm, moist soil or sand contaminated with animal feces. The larvae, most commonly from dog and cat hookworms, burrow into the skin but cannot penetrate past the outer layer. This migration causes intensely itchy, red, winding tracks on the skin, typically on the feet, legs, or buttocks.",
        "VI-chickenpox": "Chickenpox is a highly contagious viral infection caused by the varicella-zoster virus (VZV). It is characterized by an itchy rash of fluid-filled blisters that eventually scab over, accompanied by symptoms like fever and fatigue. While typically mild in children, it can cause serious complications in adults, pregnant women, and those with weakened immune systems.",
        "VI-shingles": "Shingles is a reactivation of the chickenpox virus causing painful rashes and nerve pain on one side of the body. VI, or the abducens nerve, is rarely affected by shingles, a viral infection caused by the varicella-zoster virus. This condition, a complication of herpes zoster ophthalmicus (HZO), can lead to abducens nerve palsy. The palsy causes weakness or paralysis of the lateral rectus muscle, leading to an impaired ability to move the eye outwards. Symptoms include horizontal double vision and the inability to abduct the affected eye. While recovery is common, the diplopia can persist for weeks or months after the initial rash has subsided.",
        "Actinic Keratosis" : "An actinic keratosis (ak-TIN-ik ker-uh-TOE-sis) is a rough, scaly patch on the skin that develops from years of sun exposure. It's often found on the face, lips, ears, forearms, scalp, neck or back of the hands,, increase the risk of developing skin cancer.",
        "Bowen Disease" : "Bowen disease is a pre-cancerous skin condition with a low risk of progressing to invasive squamous cell carcinoma (SCC), estimated at 3%–5%. Key risk factors for developing the condition include excessive sun exposure, fair skin, older age, and a weakened immune system. While the prognosis is generally excellent with treatment, the lesions can be progressive, and if they become invasive, one-third may potentially metastasize."
    }
    return info.get(disease_name, "Information not available for this disease.")

# --- DATA FOR HEALTH BOT (Structured Columns) ---
def disease_info_detailed(disease_name):
    details = {
        "BA- cellulitis": {
            "symptoms": ["Red, swollen, tender skin", "Area feels warm", "Pain in the area", "Possible fever", "Red streaks"],
            "medicine": ["Oral Dicloxacillin", "Cephalexin", "IV antibiotics (severe)"],
            "precautions": ["Keep skin clean", "Elevate the limb", "Avoid scratching"],
            "timetable": [
                "**Day 1-2**: Start Oral Cephalexin. Monitor for spreading redness.",
                "**Day 3-5**: Redness should stabilize; swelling starts to decrease.",
                "**Day 6-10**: Finish the entire antibiotic course to prevent relapse."
            ]
        },
        "BA-impetigo": {
            "symptoms": ["Red sores that rupture", "Honey-colored crusts", "Fluid-filled blisters"],
            "medicine": ["Mupirocin ointment", "Oral Cephalexin", "Antibacterial soap"],
            "precautions": ["Wash hands frequently", "Don't share towels", "Keep nails short"],
            "timetable": [
                "**Day 1-3**: Apply Mupirocin 2% TID. Sores start to crust.",
                "**Day 4-7**: Crusts begin to fall off. Continue topical application.",
                "**Day 8-10**: Skin heals; maintain hygiene to prevent spread."
            ]
        },
        "FU-athlete-foot": {
            "symptoms": ["Itchy, scaly red rash", "Burning sensation", "Cracking skin"],
            "medicine": ["Clotrimazole cream", "Terbinafine spray", "Antifungal powder"],
            "precautions": ["Keep feet dry", "Change socks daily", "Wear sandals in showers"],
            "timetable": [
                "**Day 1-3**: Apply Terbinafine 2x daily. Itching reduces.",
                "**Day 4-7**: Peeling skin starts to heal. Keep area very dry.",
                "**Day 8-14**: Continue treatment even if skin looks clear to kill spores."
            ]
        },
        "FU-nail-fungus": {
            "symptoms": ["Thickened nails", "Yellow-brown color", "Brittle edges"],
            "medicine": ["Oral Terbinafine", "Ciclopirox lacquer", "Laser therapy"],
            "precautions": ["Trim nails straight", "Disinfect clippers", "Moisture-wicking socks"],
            "timetable": [
                "**Day 1-10**: Start Oral Terbinafine (250mg). (Note: Treatment takes 6-12 weeks).",
                "**Month 1**: Medication reaches the nail root.",
                "**Month 3**: New healthy nail begins to push out the infected area."
            ]
        },
        "FU-ringworm": {
            "symptoms": ["Red ring-shaped rash", "Scaly border", "Itchy patches"],
            "medicine": ["Clotrimazole cream", "Miconazole", "Oral Griseofulvin"],
            "precautions": ["Avoid touching pets", "Don't share gear", "Wash bedding in hot water"],
            "timetable": [
                "**Day 1-3**: Apply antifungal cream. Border starts to flatten.",
                "**Day 4-7**: Center of the ring clears. Itching stops.",
                "**Day 8-14**: Finish the tube of cream to prevent the ring from returning."
            ]
        },
        "PA-cutaneous-larva-migrans": {
            "symptoms": ["Snake-like red tracks", "Intense nocturnal itching", "Small blisters"],
            "medicine": ["Albendazole", "Ivermectin", "Thiabendazole cream"],
            "precautions": ["Wear shoes on sand", "Deworm pets", "Use towels on beach"],
            "timetable": [
                "**Day 1-2**: Take Albendazole (400mg). Larva movement stops.",
                "**Day 3-5**: Red tracks begin to fade. Itching significantly decreases.",
                "**Day 6-10**: Skin returns to normal; tracks disappear."
            ]
        },
        "VI-chickenpox": {
            "symptoms": ["Itchy blisters", "Fever", "Fatigue"],
            "medicine": ["Acetaminophen", "Calamine lotion", "Acyclovir"],
            "precautions": ["Varicella vaccine", "Isolate from others", "Don't scratch"],
            "timetable": [
                "**Day 1-3**: Fever management. New blisters appear. Start Acyclovir if prescribed.",
                "**Day 4-6**: Blisters begin to cloud over and form scabs.",
                "**Day 7-10**: Scabs fall off. No longer contagious once all are dry."
            ]
        },
        "VI-shingles": {
            "symptoms": ["Painful one-sided rash", "Tingling", "Burning blisters"],
            "medicine": ["Acyclovir", "Valacyclovir", "Gabapentin (pain)"],
            "precautions": ["Shingles vaccine", "Keep rash covered", "Manage stress"],
            "timetable": [
                "**Day 1-3**: Start Antivirals (Acyclovir 800mg 5x/day). Pain is highest.",
                "**Day 4-7**: Blisters stop forming and begin to dry out.",
                "**Day 8-14**: Scabs form. Monitor for lingering nerve pain (PHN)."
            ]
        },
        "Actinic Keratosis": {
            "symptoms": ["Rough, scaly patch", "Wart-like surface", "Sun-exposed areas"],
            "medicine": ["Fluorouracil (5-FU)", "Imiquimod", "Cryotherapy"],
            "precautions": ["SPF 30+ daily", "Wide-brimmed hats", "Regular skin checks"],
            "timetable": [
                "**Day 1-5**: Apply 5-FU cream. Area becomes red and sore.",
                "**Day 6-14**: Peak inflammation. Skin may crust or ooze (this is expected).",
                "**Day 15+**: Treatment stops; healthy skin begins to grow back."
            ]
        },
        "Bowen Disease": {
            "symptoms": ["Red scaly persistent patch", "Slow-growing", "Crusting surface"],
            "medicine": ["Imiquimod", "5-Fluorouracil", "Surgical excision"],
            "precautions": ["Minimize UV exposure", "Monitor for changes", "High-protection SPF"],
            "timetable": [
                "**Day 1-10**: Daily application of Imiquimod. Site becomes inflamed.",
                "**Week 3-4**: Intense reaction as the cream attacks pre-cancerous cells.",
                "**Month 2**: Inflammation subsides, leaving clear skin."
            ]
        }
    }
    return details.get(disease_name, None)

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("AI MODEL FOR SKIN DISEASE TESTING")
app_mode = st.sidebar.radio(
    "Navigation", 
    ["Home", "Disease Recognition", "AI Health Bot", "Categories", "Developers Group", "About Project"]
)

# --- HOME PAGE ---
if app_mode == "Home":
    st.title("🏠 Welcome to the Skin Disease Testing System")
    st.image("sample_image.jpg", use_container_width=True)
    st.markdown("""
    ### 👋 Introduction
    Welcome to the **AI-powered Skin Disease Recognition System**.  
    This platform helps you **detect skin diseases** using deep learning.

    🔍 Upload an image, get an instant prediction, and learn about the disease.  
    """)

# --- DISEASE RECOGNITION PAGE ---
elif app_mode == "Disease Recognition":
    st.title("🧬 Disease Recognition")
    test_image = st.file_uploader("📤 Upload an Image:", type=["jpg", "jpeg", "png"])

    if test_image is not None:
        image = Image.open(test_image).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("🔎 Predict Disease"):
            with st.spinner("Model analyzing image..."):
                result_index = model_prediction(test_image)
                class_names = [
                    'BA- cellulitis', 'BA-impetigo', 'FU-athlete-foot',
                    'FU-nail-fungus', 'FU-ringworm',
                    'PA-cutaneous-larva-migrans', 'VI-chickenpox', 'VI-shingles',
                    'Actinic Keratosis', 'Bowen Disease'
                ]
                predicted_disease = class_names[result_index]
                st.success(f"🩸 The model predicts: **{predicted_disease}**")

                st.subheader("💬 AI Health Bot Summary:")
                st.info(disease_info_summary(predicted_disease))

# --- AI HEALTH BOT PAGE ---
elif app_mode == "AI Health Bot":
    st.title("🤖 Diagnose AI")
    st.markdown("""
    Type the disease name below to learn about it.  
    Example: *ringworm*, *cellulitis*, *shingles*, etc.
    """)
    query = st.text_input("Enter Disease Name:")

    if query:
        class_names = [
            'BA- cellulitis', 'BA-impetigo', 'FU-athlete-foot',
            'FU-nail-fungus', 'FU-ringworm', 'PA-cutaneous-larva-migrans',
            'VI-chickenpox', 'VI-shingles',
            'Actinic Keratosis', 'Bowen Disease'
        ]
        found = [key for key in class_names if query.lower() in key.lower()]

        if found:
            disease_name = found[0]
            data = disease_info_detailed(disease_name)

            if data:
                st.markdown(f"## Medical Analysis: {disease_name}")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.subheader("🌡️ Symptoms")
                    for s in data['symptoms']:
                        st.write(f"• {s}")

                with col2:
                    st.subheader("💊 Diagnosis & Meds")
                    for m in data['medicine']:
                        st.write(f"• {m}")

                with col3:
                    st.subheader("🛡️ Precautions")
                    for p in data['precautions']:
                        st.write(f"• {p}")
            else:
                st.warning("Detailed data is missing for this selection.")
        else:
            st.error("❌ Disease not found. Please check spelling.")

# --- CATEGORIES PAGE ---
elif app_mode == "Categories":
    st.title("📂 Disease Categories")
    st.markdown("""
    Diseases are classified into:
    - 🦠 **Bacterial** (BA): *Cellulitis*, *Impetigo*
    - 🍄 **Fungal** (FU): *Athlete's Foot*, *Ringworm*, *Nail Fungus*
    - 🪱 **Parasitic** (PA): *Cutaneous Larva Migrans*
    - 🧫 **Viral** (VI): *Chickenpox*, *Shingles*
    - ☀️ **Pre-cancerous**: *Actinic Keratosis*, *Bowen Disease*
    """)

    st.markdown("---")
    st.divider()

    st.subheader("🏥 Treatment Information and Timeline")
    user_query = st.text_input(
        "Enter the disease name to see a general recovery timeline:", 
        placeholder="e.g. Shingles, Athlete's Foot"
    )

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

            st.info(f"### 📅 General Care Timeline: {selected_disease}")

            for step in data['timetable']:
                st.markdown(f"> {step}")

            st.success(
                "💡 **Tip:** Always complete the full course of treatment as recommended by a healthcare provider, even if symptoms improve early!"
            )
        else:
            st.error("Disease not found. Please try names like 'Cellulitis', 'Ringworm', or 'Shingles'.")

# --- DEVELOPERS PAGE ---
elif app_mode == "Developers Group":
    st.title("👩‍💻 Developers Group")
    st.markdown("Meet the minds behind this project 👇")

    devs = [
        ("Dev 1", "ML Engineer", "CNN & TF Expert", "profile_image.jpeg", "https://www.linkedin.com/in/subhajit-mondal-85946328b"),
        ("Dev 2", "Frontend Dev", "Streamlit UI Expert", "dev2.jpg", "https://linkedin.com/in/user2"),
        ("Dev 3", "Data Scientist", "Data Preprocessing", "dev3.jpg", "https://linkedin.com/in/user3"),
        ("Dev 4", "Backend Dev", "API & Logic", "dev4.jpg", "https://linkedin.com/in/user4"),
        ("Dev 5", "UI Designer", "Visual Experience", "dev5.jpg", "https://linkedin.com/in/user5")
    ]

    cols = st.columns(5)
    for i, (name, role, desc, img, link) in enumerate(devs):
        with cols[i]:
            if os.path.exists(img):
                st.image(img, use_container_width=True)
            else:
                st.info("📸 Image")
            st.subheader(name)
            st.write(f"**{role}**")
            st.caption(desc)
            st.link_button("LinkedIn", link)

# --- ABOUT PAGE ---
elif app_mode == "About Project":
    st.title("ℹ️ About Project")
    st.markdown("""
    ### 🧠 Overview
    This system uses **Deep Learning (CNN)** to classify skin diseases from images.  
    **Framework:** TensorFlow + Streamlit  
    **Goal:** Build accessible healthcare AI for all 🌍
    """)

