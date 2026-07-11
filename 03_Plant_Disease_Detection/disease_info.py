"""
disease_info.py
A knowledge base mapping each class label (as produced by the model)
to human-readable information: plant, disease, description, symptoms,
treatment, and prevention. Used by the Streamlit app to show actionable
advice alongside predictions.

Keys MUST match the folder / class names used during training
(e.g. "Tomato___Early_blight").
"""

DISEASE_INFO = {

    # ---------------------------- APPLE ----------------------------
    "Apple___Apple_scab": {
        "plant": "Apple", "disease": "Apple Scab", "is_healthy": False,
        "description": "A fungal disease caused by Venturia inaequalis that produces "
                        "olive-green to black spots on leaves and fruit.",
        "symptoms": ["Olive-green/black velvety spots on leaves",
                     "Scabby, cracked lesions on fruit",
                     "Premature leaf drop"],
        "treatment": ["Apply fungicides containing captan or myclobutanil",
                      "Remove and destroy fallen leaves in autumn",
                      "Prune to improve air circulation"],
        "prevention": ["Plant scab-resistant apple varieties",
                       "Avoid overhead irrigation",
                       "Apply preventive fungicide sprays in early spring"]
    },
    "Apple___Black_rot": {
        "plant": "Apple", "disease": "Black Rot", "is_healthy": False,
        "description": "Fungal disease (Botryosphaeria obtusa) causing leaf spots, "
                        "fruit rot, and cankers on branches.",
        "symptoms": ["Purple-bordered leaf spots ('frogeye')",
                     "Concentric rings of rot on fruit", "Branch cankers"],
        "treatment": ["Remove infected fruit, leaves, and cankers",
                      "Apply fungicide (captan, thiophanate-methyl)"],
        "prevention": ["Prune out dead/diseased wood every winter",
                       "Sanitize orchard debris", "Avoid tree stress"]
    },
    "Apple___Cedar_apple_rust": {
        "plant": "Apple", "disease": "Cedar Apple Rust", "is_healthy": False,
        "description": "A fungal disease requiring both apple and cedar/juniper "
                        "hosts to complete its life cycle.",
        "symptoms": ["Bright orange/yellow spots on leaves",
                     "Tube-like structures on leaf undersides",
                     "Deformed fruit"],
        "treatment": ["Apply fungicide at pink bud stage",
                      "Remove nearby cedar/juniper galls if possible"],
        "prevention": ["Plant rust-resistant apple cultivars",
                       "Keep distance from cedar/juniper trees"]
    },
    "Apple___healthy": {
        "plant": "Apple", "disease": "Healthy", "is_healthy": True,
        "description": "No disease detected. The leaf appears healthy.",
        "symptoms": [], "treatment": [],
        "prevention": ["Continue regular monitoring",
                       "Maintain balanced fertilization and watering"]
    },

    # -------------------------- BLUEBERRY --------------------------
    "Blueberry___healthy": {
        "plant": "Blueberry", "disease": "Healthy", "is_healthy": True,
        "description": "No disease detected.",
        "symptoms": [], "treatment": [],
        "prevention": ["Maintain acidic, well-drained soil",
                       "Monitor regularly for early signs of disease"]
    },

    # ---------------------------- CHERRY ----------------------------
    "Cherry___Powdery_mildew": {
        "plant": "Cherry", "disease": "Powdery Mildew", "is_healthy": False,
        "description": "Fungal disease producing a white powdery coating on leaves "
                        "and shoots, caused by Podosphaera clandestina.",
        "symptoms": ["White powdery patches on leaves",
                     "Leaf curling and distortion", "Stunted shoot growth"],
        "treatment": ["Apply sulfur-based or potassium bicarbonate fungicides",
                      "Remove heavily infected shoots"],
        "prevention": ["Prune for better air circulation",
                       "Avoid excess nitrogen fertilization"]
    },
    "Cherry___healthy": {
        "plant": "Cherry", "disease": "Healthy", "is_healthy": True,
        "description": "No disease detected.",
        "symptoms": [], "treatment": [],
        "prevention": ["Regular pruning", "Balanced fertilization"]
    },

    # ---------------------------- CORN ----------------------------
    "Corn___Cercospora_leaf_spot Gray_leaf_spot": {
        "plant": "Corn (Maize)", "disease": "Gray Leaf Spot", "is_healthy": False,
        "description": "Fungal disease caused by Cercospora zeae-maydis, forming "
                        "rectangular gray-brown lesions on leaves.",
        "symptoms": ["Rectangular tan-to-gray lesions bound by leaf veins",
                     "Lesions merge causing leaf blight"],
        "treatment": ["Apply foliar fungicides (strobilurins/triazoles)",
                      "Rotate crops with non-host species"],
        "prevention": ["Use resistant hybrids", "Practice crop rotation",
                       "Till crop residue to reduce fungal spores"]
    },
    "Corn___Common_rust": {
        "plant": "Corn (Maize)", "disease": "Common Rust", "is_healthy": False,
        "description": "Caused by Puccinia sorghi, producing reddish-brown "
                        "pustules on both leaf surfaces.",
        "symptoms": ["Small, reddish-brown raised pustules",
                     "Pustules turn dark brown/black with age"],
        "treatment": ["Apply fungicides if severe (early growth stages)"],
        "prevention": ["Plant rust-resistant hybrids", "Early planting to avoid peak spore season"]
    },
    "Corn___Northern_Leaf_Blight": {
        "plant": "Corn (Maize)", "disease": "Northern Leaf Blight", "is_healthy": False,
        "description": "Fungal disease (Exserohilum turcicum) causing long "
                        "cigar-shaped gray-green lesions.",
        "symptoms": ["Long elliptical gray-green to tan lesions",
                     "Lesions can span several inches on leaves"],
        "treatment": ["Apply foliar fungicide at first sign of disease"],
        "prevention": ["Use resistant hybrids", "Rotate crops",
                       "Manage crop residue"]
    },
    "Corn___healthy": {
        "plant": "Corn (Maize)", "disease": "Healthy", "is_healthy": True,
        "description": "No disease detected.",
        "symptoms": [], "treatment": [],
        "prevention": ["Maintain proper spacing and nutrient balance"]
    },

    # ---------------------------- GRAPE ----------------------------
    "Grape___Black_rot": {
        "plant": "Grape", "disease": "Black Rot", "is_healthy": False,
        "description": "Fungal disease (Guignardia bidwellii) that causes "
                        "circular brown lesions and fruit mummification.",
        "symptoms": ["Small brown circular leaf spots with dark borders",
                     "Berries shrivel into hard black 'mummies'"],
        "treatment": ["Apply fungicide from bud break through veraison",
                      "Remove mummified berries and infected canes"],
        "prevention": ["Prune for canopy airflow", "Sanitize vineyard debris"]
    },
    "Grape___Esca_(Black_Measles)": {
        "plant": "Grape", "disease": "Esca (Black Measles)", "is_healthy": False,
        "description": "A complex fungal trunk disease causing tiger-stripe "
                        "leaf patterns and berry spotting.",
        "symptoms": ["Tiger-stripe interveinal leaf discoloration",
                     "Dark spots on berries", "Wood decay in trunk"],
        "treatment": ["No full cure; remove and destroy infected vines",
                      "Apply trunk surgery in early cases"],
        "prevention": ["Avoid large pruning wounds",
                       "Protect pruning cuts with sealant"]
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "plant": "Grape", "disease": "Leaf Blight (Isariopsis Leaf Spot)", "is_healthy": False,
        "description": "Fungal disease causing irregular brown spots leading to "
                        "premature defoliation.",
        "symptoms": ["Irregular reddish-brown leaf spots",
                     "Yellowing around lesions", "Early leaf drop"],
        "treatment": ["Apply copper-based or systemic fungicides"],
        "prevention": ["Improve air circulation", "Remove fallen infected leaves"]
    },
    "Grape___healthy": {
        "plant": "Grape", "disease": "Healthy", "is_healthy": True,
        "description": "No disease detected.",
        "symptoms": [], "treatment": [],
        "prevention": ["Regular canopy management", "Balanced irrigation"]
    },

    # ---------------------------- ORANGE ----------------------------
    "Orange___Haunglongbing_(Citrus_greening)": {
        "plant": "Orange", "disease": "Citrus Greening (HLB)", "is_healthy": False,
        "description": "A severe bacterial disease spread by psyllid insects, "
                        "with no known cure.",
        "symptoms": ["Blotchy yellow mottling on leaves",
                     "Small, lopsided, bitter fruit",
                     "Tree decline and dieback"],
        "treatment": ["Remove and destroy infected trees",
                      "Control psyllid vector with insecticides"],
        "prevention": ["Use certified disease-free nursery stock",
                       "Regular psyllid monitoring and control"]
    },

    # ---------------------------- PEACH ----------------------------
    "Peach___Bacterial_spot": {
        "plant": "Peach", "disease": "Bacterial Spot", "is_healthy": False,
        "description": "Caused by Xanthomonas campestris, producing small "
                        "dark lesions on leaves and fruit.",
        "symptoms": ["Small angular water-soaked leaf spots",
                     "Spots may fall out, leaving 'shot holes'",
                     "Sunken lesions on fruit"],
        "treatment": ["Apply copper-based bactericides during dormancy"],
        "prevention": ["Plant resistant varieties", "Avoid overhead irrigation"]
    },
    "Peach___healthy": {
        "plant": "Peach", "disease": "Healthy", "is_healthy": True,
        "description": "No disease detected.",
        "symptoms": [], "treatment": [],
        "prevention": ["Regular pruning and monitoring"]
    },

    # -------------------------- PEPPER BELL --------------------------
    "Pepper_bell___Bacterial_spot": {
        "plant": "Bell Pepper", "disease": "Bacterial Spot", "is_healthy": False,
        "description": "Caused by Xanthomonas spp., producing water-soaked "
                        "spots that turn brown and scabby.",
        "symptoms": ["Small water-soaked leaf spots turning dark brown",
                     "Raised, scabby spots on fruit"],
        "treatment": ["Apply copper-based bactericide sprays",
                      "Remove and destroy infected plants"],
        "prevention": ["Use disease-free seeds", "Avoid working in wet fields",
                       "Rotate crops"]
    },
    "Pepper_bell___healthy": {
        "plant": "Bell Pepper", "disease": "Healthy", "is_healthy": True,
        "description": "No disease detected.",
        "symptoms": [], "treatment": [],
        "prevention": ["Maintain proper spacing and drip irrigation"]
    },

    # ---------------------------- POTATO ----------------------------
    "Potato___Early_blight": {
        "plant": "Potato", "disease": "Early Blight", "is_healthy": False,
        "description": "Fungal disease (Alternaria solani) forming concentric "
                        "'target-spot' rings on older leaves.",
        "symptoms": ["Dark brown concentric ring spots",
                     "Yellowing around lesions", "Lower leaves affected first"],
        "treatment": ["Apply fungicide (chlorothalonil, mancozeb)",
                      "Remove infected foliage"],
        "prevention": ["Rotate crops (2-3 years)", "Ensure adequate plant nutrition",
                       "Avoid overhead watering"]
    },
    "Potato___Late_blight": {
        "plant": "Potato", "disease": "Late Blight", "is_healthy": False,
        "description": "Caused by Phytophthora infestans (the Irish Potato "
                        "Famine pathogen); spreads rapidly in cool, wet weather.",
        "symptoms": ["Water-soaked gray-green lesions",
                     "White fungal growth on leaf undersides",
                     "Rapid blackening and plant collapse"],
        "treatment": ["Apply systemic fungicides immediately (e.g., metalaxyl)",
                      "Destroy infected plants to stop spread"],
        "prevention": ["Plant certified disease-free seed potatoes",
                       "Improve field drainage and airflow",
                       "Monitor weather-based disease forecasts"]
    },
    "Potato___healthy": {
        "plant": "Potato", "disease": "Healthy", "is_healthy": True,
        "description": "No disease detected.",
        "symptoms": [], "treatment": [],
        "prevention": ["Practice crop rotation", "Use certified seed potatoes"]
    },

    # --------------------------- SOYBEAN ---------------------------
    "Soybean___healthy": {
        "plant": "Soybean", "disease": "Healthy", "is_healthy": True,
        "description": "No disease detected.",
        "symptoms": [], "treatment": [],
        "prevention": ["Rotate crops", "Monitor for pest pressure regularly"]
    },

    # -------------------------- STRAWBERRY --------------------------
    "Strawberry___Leaf_scorch": {
        "plant": "Strawberry", "disease": "Leaf Scorch", "is_healthy": False,
        "description": "Fungal disease (Diplocarpon earlianum) causing purple "
                        "spots that merge into scorched-looking leaves.",
        "symptoms": ["Small purple spots on upper leaf surface",
                     "Spots merge, leaves appear scorched/dried"],
        "treatment": ["Apply fungicide (captan, myclobutanil)",
                      "Remove infected leaves after harvest"],
        "prevention": ["Use resistant varieties", "Avoid overhead irrigation",
                       "Space plants for airflow"]
    },
    "Strawberry___healthy": {
        "plant": "Strawberry", "disease": "Healthy", "is_healthy": True,
        "description": "No disease detected.",
        "symptoms": [], "treatment": [],
        "prevention": ["Mulch to reduce soil splash", "Regular monitoring"]
    },

    # ---------------------------- TOMATO ----------------------------
    "Tomato___Bacterial_spot": {
        "plant": "Tomato", "disease": "Bacterial Spot", "is_healthy": False,
        "description": "Caused by Xanthomonas spp., producing small dark "
                        "greasy spots on leaves and fruit.",
        "symptoms": ["Small water-soaked spots that turn dark brown",
                     "Spots on fruit are raised and scabby"],
        "treatment": ["Apply copper-based bactericides",
                      "Remove and destroy infected plant debris"],
        "prevention": ["Use certified disease-free seed",
                       "Avoid overhead watering", "Rotate crops"]
    },
    "Tomato___Early_blight": {
        "plant": "Tomato", "disease": "Early Blight", "is_healthy": False,
        "description": "Fungal disease (Alternaria solani) forming concentric "
                        "target-like rings, usually on older leaves first.",
        "symptoms": ["Dark concentric ring spots on lower leaves",
                     "Yellow halo around lesions", "Leaf drop over time"],
        "treatment": ["Apply fungicide (chlorothalonil, mancozeb)",
                      "Remove infected lower leaves"],
        "prevention": ["Mulch soil to prevent splash",
                       "Stake plants for airflow", "Rotate crops annually"]
    },
    "Tomato___Late_blight": {
        "plant": "Tomato", "disease": "Late Blight", "is_healthy": False,
        "description": "Caused by Phytophthora infestans; can destroy a crop "
                        "within days under cool, humid conditions.",
        "symptoms": ["Water-soaked lesions that turn brown/black rapidly",
                     "White mold on leaf undersides in humid weather"],
        "treatment": ["Apply systemic fungicide immediately",
                      "Remove and destroy infected plants"],
        "prevention": ["Avoid overhead irrigation", "Ensure good air circulation",
                       "Monitor local blight forecasts"]
    },
    "Tomato___Leaf_Mold": {
        "plant": "Tomato", "disease": "Leaf Mold", "is_healthy": False,
        "description": "Fungal disease (Passalora fulva) common in humid "
                        "greenhouse conditions.",
        "symptoms": ["Pale yellow spots on upper leaf surface",
                     "Olive-green/velvety mold on leaf undersides"],
        "treatment": ["Apply fungicide", "Improve ventilation to reduce humidity"],
        "prevention": ["Use resistant varieties", "Reduce greenhouse humidity",
                       "Increase plant spacing"]
    },
    "Tomato___Septoria_leaf_spot": {
        "plant": "Tomato", "disease": "Septoria Leaf Spot", "is_healthy": False,
        "description": "Fungal disease (Septoria lycopersici) causing many "
                        "small circular spots with dark borders.",
        "symptoms": ["Numerous small circular spots with gray centers",
                     "Severe leaf yellowing and drop"],
        "treatment": ["Apply fungicide (chlorothalonil)",
                      "Remove infected leaves promptly"],
        "prevention": ["Rotate crops", "Avoid wetting foliage",
                       "Clean up plant debris after season"]
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "plant": "Tomato", "disease": "Two-Spotted Spider Mite", "is_healthy": False,
        "description": "A pest infestation (Tetranychus urticae) rather than a "
                        "pathogen, causing stippled, bronzed leaves.",
        "symptoms": ["Fine yellow stippling on leaves",
                     "Fine webbing on leaf undersides", "Bronze leaf discoloration"],
        "treatment": ["Apply miticide or insecticidal soap",
                      "Introduce predatory mites (biological control)"],
        "prevention": ["Keep plants well-watered (mites thrive in drought stress)",
                       "Regularly inspect leaf undersides"]
    },
    "Tomato___Target_Spot": {
        "plant": "Tomato", "disease": "Target Spot", "is_healthy": False,
        "description": "Fungal disease (Corynespora cassiicola) producing "
                        "concentric target-like lesions.",
        "symptoms": ["Brown concentric-ring lesions on leaves and fruit",
                     "Lesions may coalesce causing defoliation"],
        "treatment": ["Apply fungicide (azoxystrobin, chlorothalonil)"],
        "prevention": ["Improve airflow via pruning/staking",
                       "Rotate crops", "Remove crop debris"]
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "plant": "Tomato", "disease": "Yellow Leaf Curl Virus", "is_healthy": False,
        "description": "A viral disease transmitted by whiteflies, causing "
                        "severe stunting and leaf curling.",
        "symptoms": ["Upward curling, yellowing leaves",
                     "Stunted plant growth", "Reduced fruit set"],
        "treatment": ["No cure; remove and destroy infected plants",
                      "Control whitefly populations with insecticide"],
        "prevention": ["Use resistant tomato varieties",
                       "Use reflective mulch to repel whiteflies",
                       "Install insect netting in nurseries"]
    },
    "Tomato___Tomato_mosaic_virus": {
        "plant": "Tomato", "disease": "Mosaic Virus", "is_healthy": False,
        "description": "A viral disease causing mottled light/dark green "
                        "patterns on leaves and reduced yield.",
        "symptoms": ["Mottled green/yellow mosaic pattern on leaves",
                     "Leaf curling and fern-like distortion",
                     "Stunted growth, reduced fruit quality"],
        "treatment": ["No cure; remove and destroy infected plants",
                      "Disinfect tools between plants"],
        "prevention": ["Use certified virus-free seed",
                       "Wash hands/tools between handling plants",
                       "Control aphid vectors"]
    },
    "Tomato___healthy": {
        "plant": "Tomato", "disease": "Healthy", "is_healthy": True,
        "description": "No disease detected. The leaf appears healthy.",
        "symptoms": [], "treatment": [],
        "prevention": ["Continue regular monitoring",
                       "Maintain balanced watering and fertilization"]
    },
}


def get_disease_info(class_name: str) -> dict:
    """
    Safely fetch disease info for a predicted class name.
    Falls back to a generic 'unknown' entry if the class isn't in the DB.
    """
    return DISEASE_INFO.get(class_name, {
        "plant": class_name.split("___")[0].replace("_", " ") if "___" in class_name else "Unknown",
        "disease": class_name.split("___")[-1].replace("_", " ") if "___" in class_name else class_name,
        "is_healthy": "healthy" in class_name.lower(),
        "description": "Detailed information for this class is not yet available in our database.",
        "symptoms": [],
        "treatment": ["Consult a local agricultural extension officer for confirmation and treatment advice."],
        "prevention": ["Practice good field sanitation and crop rotation."]
    })