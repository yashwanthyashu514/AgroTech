"""
Raithanna (ರೈತಣ್ಣ) Plugin Configuration.
Defines persona, bilingual greetings, and farmer quick actions.
"""
from typing import Dict, List, Any

BOT_NAME_KN = "ರೈತಣ್ಣ"
BOT_NAME_EN = "Raithanna"
BOT_NAME_HI = "रायतण्णा"
TAGLINE_KN = "ಕರ್ನಾಟಕದ ರೈತರ ನೆಚ್ಚಿನ AI ಕೃಷಿ ಮಿತ್ರ"
TAGLINE_EN = "Karnataka's Trusted AI Farmer Companion"
TAGLINE_HI = "कर्नाटक के किसानों का भरोसेमंद AI कृषि साथी"

GREETING_KN = (
    "ನಮಸ್ಕಾರ! ನಿಮ್ಮ AI ಕೃಷಿ ಸಹಾಯಕ್ಕೆ ಸುಸ್ವಾಗತ. 🙏\n"
    "ನಾನು ಸಸ್ಯಗಳನ್ನು ಗುರುತಿಸಲು, ಬೆಳೆ ರೋಗಗಳನ್ನು ಪತ್ತೆಹಚ್ಚಲು, ಮಣ್ಣಿನ ವರದಿಗಳನ್ನು ವಿಶ್ಲೇಷಿಸಲು ಮತ್ತು ನಿಮ್ಮ ಕೃಷಿ ಪ್ರಶ್ನೆಗಳಿಗೆ ಉತ್ತರಿಸಲು ಸಹಾಯ ಮಾಡುತ್ತೇನೆ.\n"
    "ಕನ್ನಡ, ಇಂಗ್ಲಿಷ್ ಅಥವಾ ಹಿಂದಿಯಲ್ಲಿ ಮಾತನಾಡಬಹುದು (🎙️) ಅಥವಾ ಫೋಟೋ ಅಪ್ಲೋಡ್ ಮಾಡಬಹುದು (📷)!"
)

GREETING_EN = (
    "Namaskara! Welcome to your Agriculture AI Assistant. 🙏\n"
    "I can help you identify plants, diagnose possible crop diseases, analyse soil reports and answer agricultural questions.\n"
    "You can speak via voice (🎙️) or upload plant and soil photographs (📷) in Kannada, English or Hindi!"
)

GREETING_HI = (
    "नमस्कार! आपके AI कृषि सहायक में आपका स्वागत है। 🙏\n"
    "मैं पौधों की पहचान करने, फसल रोगों का निदान करने, मिट्टी की रिपोर्ट का विश्लेषण करने और आपके कृषि प्रश्नों के उत्तर देने में मदद कर सकता हूँ।\n"
    "आप बोलकर (🎙️) या फोटो भेजकर (📷) कन्नड़, अंग्रेजी या हिंदी में पूछ सकते हैं!"
)

QUICK_ACTIONS: List[Dict[str, Any]] = [
    {
        "id": "identify_plant",
        "icon": "📷",
        "title_en": "Identify plant",
        "title_kn": "ಸಸ್ಯ ಗುರುತಿಸಿ",
        "title_hi": "पौधे की पहचान",
        "action_type": "camera",
        "prompt_en": "I want to identify this plant or crop. Please tell me its species, common name, and care requirements.",
        "prompt_kn": "ಈ ಸಸ್ಯ ಅಥವಾ ಬೆಳೆ ಯಾವುದು ಎಂದು ಗುರುತಿಸಿ. ಇದರ ಹೆಸರು ಮತ್ತು ಬೆಳೆಯುವ ವಿಧಾನ ತಿಳಿಸಿ.",
        "prompt_hi": "कृपया इस पौधे या फसल की पहचान करें और इसका नाम तथा देखरेख के तरीके बताएं।",
    },
    {
        "id": "detect_disease",
        "icon": "🔍",
        "title_en": "Detect disease",
        "title_kn": "ರೋಗ ಪತ್ತೆಹಚ್ಚಿ",
        "title_hi": "रोग का पता लगाएं",
        "action_type": "camera",
        "prompt_en": "My crop leaves have spots/curling. What disease is this and what is the organic/chemical treatment?",
        "prompt_kn": "ನನ್ನ ಬೆಳೆಯ ಎಲೆಗಳಲ್ಲಿ ಕಲೆಗಳು/ಹಳದಿಯಾಗುತ್ತಿದೆ. ಇದು ಯಾವ ರೋಗ ಮತ್ತು ಇದಕ್ಕೆ ಸರಿಯಾದ ಔಷಧ ಯಾವುದು?",
        "prompt_hi": "मेरी फसल की पत्तियों पर धब्बे हैं। यह कौन सा रोग है और इसका सही उपचार क्या है?",
    },
    {
        "id": "analyse_soil",
        "icon": "🌱",
        "title_en": "Analyse soil",
        "title_kn": "ಮಣ್ಣು ವಿಶ್ಲೇಷಣೆ",
        "title_hi": "मिट्टी विश्लेषण",
        "action_type": "camera_or_file",
        "prompt_en": "Analyse this soil photo or soil health card report. What are the best crops, pH, and fertilizer doses?",
        "prompt_kn": "ಈ ಮಣ್ಣಿನ ಫೋಟೋ ಅಥವಾ ಮಣ್ಣಿನ ಆರೋಗ್ಯ ಕಾರ್ಡ್ ವಿಶ್ಲೇಷಿಸಿ. ಇದಕ್ಕೆ ಸೂಕ್ತವಾದ ಬೆಳೆಗಳು ಮತ್ತು ಗೊಬ್ಬರದ ಪ್ರಮಾಣ ತಿಳಿಸಿ.",
        "prompt_hi": "इस मिट्टी की फोटो या रिपोर्ट का विश्लेषण करें। इसके लिए सबसे अच्छी फसलें और खाद की मात्रा बताएं।",
    },
    {
        "id": "identify_pests",
        "icon": "🐛",
        "title_en": "Identify pests",
        "title_kn": "ಕೀಟ ಗುರುತಿಸಿ",
        "title_hi": "कीट पहचानें",
        "action_type": "camera",
        "prompt_en": "There are insects/worms attacking my crop. Identify the pest and suggest safe IPM control measures.",
        "prompt_kn": "ನನ್ನ ಬೆಳೆಗೆ ಕೀಟಗಳು ಅಥವಾ ಹುಳುಗಳ ಬಾಧೆ ಇದೆ. ಈ ಕೀಟವನ್ನು ಗುರುತಿಸಿ ಮತ್ತು ನಿಯಂತ್ರಣ ಕ್ರಮಗಳನ್ನು ತಿಳಿಸಿ.",
        "prompt_hi": "मेरी फसल पर कीड़े या कीट का हमला हुआ है। इस कीट की पहचान करें और रोकथाम के उपाय बताएं।",
    },
    {
        "id": "weather_advice",
        "icon": "🌦️",
        "title_en": "Weather advice",
        "title_kn": "ಹವಾಮಾನ ಸಲಹೆ",
        "title_hi": "मौसम सलाह",
        "action_type": "chat",
        "prompt_en": "What is the 7-day weather forecast, rainfall probability, and farming advisory for my district?",
        "prompt_kn": "ನನ್ನ ಜಿಲ್ಲೆಗೆ ಮುಂದಿನ 7 ದಿನಗಳ ಹವಾಮಾನ ಮುನ್ಸೂಚನೆ, ಮಳೆ ಮತ್ತು ಕೃಷಿ ಸಲಹೆ ಹೇಗಿದೆ?",
        "prompt_hi": "मेरे जिले के लिए अगले 7 दिनों का मौसम पूर्वानुमान, बारिश की संभावना और कृषि सलाह क्या है?",
    },
    {
        "id": "ask_ai",
        "icon": "💬",
        "title_en": "Ask agriculture AI",
        "title_kn": "ಕೃಷಿ AI ಪ್ರಶ್ನಿಸಿ",
        "title_hi": "कृषि AI से पूछें",
        "action_type": "chat",
        "prompt_en": "What government schemes, APMC mandi rates, or crop management practices should I follow this season?",
        "prompt_kn": "ಈ ಋತುವಿನಲ್ಲಿ ಕರ್ನಾಟಕದ ಪ್ರಮುಖ ಸರ್ಕಾರಿ ಸಬ್ಸಿಡಿಗಳು, ಮಾರುಕಟ್ಟೆ ದರಗಳು ಮತ್ತು ಕೃಷಿ ಯೋಜನೆಗಳೇನು?",
        "prompt_hi": "इस मौसम में कर्नाटक की प्रमुख सरकारी योजनाएं, मंडी भाव और खेती के सर्वोत्तम तरीके क्या हैं?",
    },
]
