import React, { useState, useEffect, useRef } from 'react';

const KARNATAKA_DISTRICTS = [
  'Bagalkot', 'Ballari', 'Belagavi', 'Bengaluru Rural', 'Bengaluru Urban',
  'Bidar', 'Chamarajanagar', 'Chikkaballapura', 'Chikkamagaluru', 'Chitradurga',
  'Dakshina Kannada', 'Davanagere', 'Dharwad', 'Gadag', 'Hassan',
  'Haveri', 'Kalaburagi', 'Kodagu', 'Kolar', 'Koppal',
  'Mandya', 'Mysuru', 'Raichur', 'Ramanagara', 'Shivamogga',
  'Tumakuru', 'Udupi', 'Uttara Kannada', 'Vijayapura', 'Yadgir'
];

export function AIAssistantPage() {
  const [lang, setLang] = useState('kn'); // 'kn' | 'en' | 'hi'
  const [district, setDistrict] = useState('Davanagere');
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [activeFeature, setActiveFeature] = useState(null);
  const [recording, setRecording] = useState(false);
  const [speakingIdx, setSpeakingIdx] = useState(null);

  // Soil modal state
  const [showSoilModal, setShowSoilModal] = useState(false);
  const [soilForm, setSoilForm] = useState({ ph: '6.8', n: '240', p: '22', k: '190', oc: '0.45' });

  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);
  const recognitionRef = useRef(null);

  // Trilingual greetings
  const greetings = {
    kn: {
      title: 'ನಮಸ್ಕಾರ! ನಿಮ್ಮ AI ಕೃಷಿ ಸಹಾಯಕ್ಕೆ ಸುಸ್ವಾಗತ.',
      sub: 'ನಾನು ಸಸ್ಯಗಳನ್ನು ಗುರುತಿಸಲು, ಬೆಳೆ ರೋಗಗಳನ್ನು ಪತ್ತೆಹಚ್ಚಲು, ಮಣ್ಣಿನ ವರದಿಗಳನ್ನು ವಿಶ್ಲೇಷಿಸಲು ಮತ್ತು ನಿಮ್ಮ ಕೃಷಿ ಪ್ರಶ್ನೆಗಳಿಗೆ ಉತ್ತರಿಸಲು ಸಹಾಯ ಮಾಡುತ್ತೇನೆ.',
      placeholder: 'ನಿಮ್ಮ ಕೃಷಿ ಪ್ರಶ್ನೆ ಕೇಳಿ...',
    },
    en: {
      title: 'Namaskara! Welcome to your Agriculture AI Assistant.',
      sub: 'I can help you identify plants, diagnose possible crop diseases, analyse soil reports and answer agricultural questions.',
      placeholder: 'Ask your agriculture question...',
    },
    hi: {
      title: 'नमस्कार! आपके AI कृषि सहायक में आपका स्वागत है।',
      sub: 'मैं पौधों की पहचान करने, फसल रोगों का निदान करने, मिट्टी की रिपोर्ट का विश्लेषण करने और आपके कृषि प्रश्नों के उत्तर देने में मदद कर सकता हूँ।',
      placeholder: 'अपना कृषि प्रश्न पूछें...',
    }
  };

  const featureCards = [
    {
      id: 'identify_plant',
      icon: '📷',
      title: { kn: 'ಸಸ್ಯ ಗುರುತಿಸಿ', en: 'Identify plant', hi: 'पौधे पहचानें' },
      desc: { kn: 'ಫೋಟೋ ಮೂಲಕ ಬೆಳೆ ಮತ್ತು ಜಾತಿ ಪತ್ತೆ', en: 'Identify crop & plant species from photo', hi: 'फोटो से पौधे व फसल की पहचान' },
      type: 'photo'
    },
    {
      id: 'detect_disease',
      icon: '🔍',
      title: { kn: 'ರೋಗ ಪತ್ತೆಹಚ್ಚಿ', en: 'Detect disease', hi: 'रोग का पता लगाएं' },
      desc: { kn: 'ಎಲೆ ಕಲೆ, ಕರಕಲು, ಬ್ಲೈಟ್ ಪರೀಕ್ಷೆ & ಔಷಧ', en: 'Analyse leaf symptoms & get treatment', hi: 'पत्तियों के रोग की जांच व उपचार' },
      type: 'photo'
    },
    {
      id: 'analyse_soil',
      icon: '🌱',
      title: { kn: 'ಮಣ್ಣು ವಿಶ್ಲೇಷಣೆ', en: 'Analyse soil', hi: 'मिट्टी विश्लेषण' },
      desc: { kn: 'ಮಣ್ಣಿನ ಫೋಟೋ ಅಥವಾ ಮಣ್ಣು ಪರೀಕ್ಷೆ ವರದಿ', en: 'Soil image classification & SHC report', hi: 'मिट्टी फोटो या सॉइल टेस्ट कार्ड' },
      type: 'soil'
    },
    {
      id: 'identify_pests',
      icon: '🐛',
      title: { kn: 'ಕೀಟ ಗುರುತಿಸಿ', en: 'Identify pests', hi: 'कीट पहचानें' },
      desc: { kn: 'ಹುಳು, ನುಸಿ, ಲದ್ದಿಹುಳು & IPM ನಿಯಂತ್ರಣ', en: 'Insect pest detection & IPM remedies', hi: 'कीटों की पहचान व रोकथाम उपाय' },
      type: 'photo'
    },
    {
      id: 'weather_advice',
      icon: '🌦️',
      title: { kn: 'ಹವಾಮಾನ ಸಲಹೆ', en: 'Weather advice', hi: 'मौसम सलाह' },
      desc: { kn: '7 ದಿನಗಳ ಮಳೆ ಮುನ್ಸೂಚನೆ & ಕೃಷಿ ಸಲಹೆ', en: '7-day forecast & farming advisory', hi: '7 दिनों का मौसम व कृषि सलाह' },
      type: 'action'
    },
    {
      id: 'ask_ai',
      icon: '💬',
      title: { kn: 'ಕೃಷಿ AI ಪ್ರಶ್ನಿಸಿ', en: 'Ask agriculture AI', hi: 'कृषि AI से पूछें' },
      desc: { kn: 'ಸಬ್ಸಿಡಿ, ಮಾರುಕಟ್ಟೆ ದರ, ಕೃಷಿ ಪ್ರಶ್ನೋತ್ತರ', en: 'Govt schemes, APMC prices & crop tips', hi: 'सरकारी योजनाएं, मंडी भाव व सलाह' },
      type: 'chat'
    }
  ];

  // Initialize Speech Recognition
  useEffect(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
      const recognition = new SpeechRecognition();
      recognition.continuous = false;
      recognition.interimResults = false;
      recognition.lang = lang === 'kn' ? 'kn-IN' : (lang === 'hi' ? 'hi-IN' : 'en-IN');
      recognition.onresult = (e) => {
        const text = e.results[0][0].transcript;
        setInput(text);
        setRecording(false);
        handleSend(text);
      };
      recognition.onerror = () => setRecording(false);
      recognition.onend = () => setRecording(false);
      recognitionRef.current = recognition;
    }
  }, [lang]);

  // Set welcome message on load or language change
  useEffect(() => {
    const welcomeText = lang === 'kn' 
      ? `🙏 ನಮಸ್ಕಾರ! ನಾನು ನಿಮ್ಮ ರೈತಣ್ಣ (Raithanna) - ಕರ್ನಾಟಕ ಕೃಷಿ AI ಸಹಾಯಕ.\nಬೆಳೆ ರೋಗ, ಸಸ್ಯ ಗುರುತಿಸುವಿಕೆ, ಮಣ್ಣು ಪರೀಕ್ಷೆ, ಇಂದಿನ ಹವಾಮಾನ (${district}) ಅಥವಾ ಸರ್ಕಾರಿ ಯೋಜನೆಗಳ ಕುರಿತು ಯಾವುದೇ ಪ್ರಶ್ನೆ ಕೇಳಿ ಅಥವಾ ಫೋಟೋ ಅಪ್ಲೋಡ್ ಮಾಡಿ.`
      : lang === 'hi'
      ? `🙏 नमस्कार! मैं आपका कृषि AI सहायक रायतण्णा हूँ।\nफसल रोग, पौधे की पहचान, मिट्टी परीक्षण, मौसम पूर्वानुमान (${district}) या सरकारी योजनाओं के बारे में कोई भी प्रश्न पूछें या फोटो अपलोड करें।`
      : `🙏 Namaskara! I am your Raithanna - Karnataka Agriculture AI Assistant.\nAsk any farming question, check weather (${district}), or upload plant and soil photos for instant diagnosis and treatment guidance.`;

    setMessages([
      {
        id: 'welcome',
        sender: 'bot',
        text: welcomeText,
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        chips: lang === 'kn' 
          ? ['ಇಂದಿನ ಹವಾಮಾನ', 'ಬೆಳೆ ರೋಗ ಪರೀಕ್ಷೆ', 'ಸರ್ಕಾರಿ ಯೋಜನೆಗಳು', 'ಮಾರುಕಟ್ಟೆ ಎಪಿಎಂಸಿ ದರ']
          : ['Today Weather', 'Leaf Disease Check', 'Govt Schemes', 'APMC Mandi Rates']
      }
    ]);
  }, [lang, district]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const toggleVoiceRecording = () => {
    if (!recognitionRef.current) {
      alert(lang === 'kn' ? 'ಧ್ವನಿ ಗುರುತಿಸುವಿಕೆ ಬೆಂಬಲಿತವಾಗಿಲ್ಲ. ದಯವಿಟ್ಟು ಟೈಪ್ ಮಾಡಿ.' : 'Speech recognition not supported in this browser. Please type.');
      return;
    }
    if (recording) {
      recognitionRef.current.stop();
      setRecording(false);
    } else {
      try {
        recognitionRef.current.lang = lang === 'kn' ? 'kn-IN' : (lang === 'hi' ? 'hi-IN' : 'en-IN');
        recognitionRef.current.start();
        setRecording(true);
      } catch (err) {
        setRecording(false);
      }
    }
  };

  const speakText = (text, idx) => {
    if (!window.speechSynthesis) return;
    if (speakingIdx === idx) {
      window.speechSynthesis.cancel();
      setSpeakingIdx(null);
      return;
    }
    window.speechSynthesis.cancel();
    const cleanText = text.replace(/[*_#`~[\]]/g, '').replace(/https?:\/\/\S+/g, '');
    const utterance = new SpeechSynthesisUtterance(cleanText);
    utterance.lang = lang === 'kn' ? 'kn-IN' : (lang === 'hi' ? 'hi-IN' : 'en-IN');
    utterance.rate = 0.95;
    utterance.onend = () => setSpeakingIdx(null);
    utterance.onerror = () => setSpeakingIdx(null);
    setSpeakingIdx(idx);
    window.speechSynthesis.speak(utterance);
  };

  const renderMessageContent = (rawText) => {
    if (!rawText) return null;
    const lines = rawText.split('\n');
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
        {lines.map((line, lIdx) => {
          if (!line.trim()) {
            return <div key={lIdx} style={{ height: '6px' }} />;
          }
          // Highlight [BRACKET HEADERS] cleanly
          if (line.startsWith('[') && line.includes(']:')) {
            const closingIdx = line.indexOf(']:');
            const header = line.substring(0, closingIdx + 2);
            const content = line.substring(closingIdx + 2);
            return (
              <div key={lIdx} style={{ lineHeight: 1.5 }}>
                <span style={{ fontWeight: 700, color: '#86efac', marginRight: '6px' }}>{header}</span>
                <span style={{ color: '#f8fafc' }}>{content}</span>
              </div>
            );
          }
          // Parse any **bold** text without leaking asterisks
          const parts = line.split(/(\*\*.*?\*\*)/g);
          return (
            <div key={lIdx} style={{ lineHeight: 1.5 }}>
              {parts.map((part, pIdx) => {
                if (part.startsWith('**') && part.endsWith('**') && part.length > 4) {
                  return <strong key={pIdx} style={{ color: '#86efac', fontWeight: 700 }}>{part.slice(2, -2)}</strong>;
                }
                return <span key={pIdx}>{part}</span>;
              })}
            </div>
          );
        })}
      </div>
    );
  };

  const handleCardClick = (card) => {
    setActiveFeature(card.id);
    if (card.type === 'photo') {
      fileInputRef.current?.click();
    } else if (card.type === 'soil') {
      setShowSoilModal(true);
    } else if (card.id === 'weather_advice') {
      executeFeatureRequest('weather_advice', null, `What is the 7-day weather forecast and agromet advisory for ${district}?`);
    } else {
      const prompt = lang === 'kn' 
        ? `${card.title.kn} ಕುರಿತು ಕರ್ನಾಟಕದ ಪ್ರಸ್ತುತ ಮಾಹಿತಿ ತಿಳಿಸಿ.`
        : `${card.title.en} advice for ${district} district, Karnataka.`;
      handleSend(prompt);
    }
  };

  const handleFileUpload = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const feature = activeFeature || 'detect_disease';

    const userMsg = {
      id: Date.now(),
      sender: 'user',
      text: `📷 [${feature.toUpperCase()}] ${file.name}`,
      previewUrl: URL.createObjectURL(file),
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };
    setMessages(prev => [...prev, userMsg]);
    executeFeatureRequest(feature, file, null);
    e.target.value = '';
  };

  const executeFeatureRequest = async (featureId, file, customMsg) => {
    setLoading(true);
    const formData = new FormData();
    formData.append('feature_id', featureId);
    formData.append('district', district);
    formData.append('lang', lang);
    if (file) formData.append('file', file);
    if (customMsg) formData.append('message', customMsg);

    try {
      const res = await fetch('/api/v1/plugin/raithanna/feature-action', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      setMessages(prev => [
        ...prev,
        {
          id: Date.now() + 1,
          sender: 'bot',
          text: data.reply,
          chips: data.chips || [],
          time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        }
      ]);
    } catch (err) {
      setMessages(prev => [
        ...prev,
        {
          id: Date.now() + 1,
          sender: 'bot',
          text: lang === 'kn' ? 'ಕ್ಷಮಿಸಿ, ವಿನಂತಿ ಪ್ರಕ್ರಿಯೆಯಲ್ಲಿ ದೋಷವಾಗಿದೆ. ದಯವಿಟ್ಟು ಪುನಃ ಪ್ರಯತ್ನಿಸಿ.' : 'Request processing error. Please try again.',
          time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        }
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleSoilSubmit = () => {
    setShowSoilModal(false);
    const userMsg = {
      id: Date.now(),
      sender: 'user',
      text: `🧪 Soil Lab Report: pH=${soilForm.ph}, N=${soilForm.n}, P=${soilForm.p}, K=${soilForm.k}, OC=${soilForm.oc}%`,
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };
    setMessages(prev => [...prev, userMsg]);
    setLoading(true);

    const formData = new FormData();
    formData.append('feature_id', 'analyse_soil');
    formData.append('district', district);
    formData.append('lang', lang);
    formData.append('soil_ph', soilForm.ph);
    formData.append('soil_n', soilForm.n);
    formData.append('soil_p', soilForm.p);
    formData.append('soil_k', soilForm.k);
    formData.append('soil_oc', soilForm.oc);

    fetch('/api/v1/plugin/raithanna/feature-action', { method: 'POST', body: formData })
      .then(r => r.json())
      .then(data => {
        setMessages(prev => [
          ...prev,
          {
            id: Date.now() + 1,
            sender: 'bot',
            text: data.reply,
            chips: data.chips || [],
            time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
          }
        ]);
      })
      .catch(() => {
        setMessages(prev => [...prev, { id: Date.now() + 1, sender: 'bot', text: 'Soil analysis error.', time: 'now' }]);
      })
      .finally(() => setLoading(false));
  };

  const handleSend = async (customText) => {
    const textToSend = customText || input;
    if (!textToSend.trim()) return;

    const userMsg = {
      id: Date.now(),
      sender: 'user',
      text: textToSend,
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    try {
      const res = await fetch('/api/v1/plugin/raithanna/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: textToSend, district, lang })
      });
      const data = await res.json();
      setMessages(prev => [
        ...prev,
        {
          id: Date.now() + 1,
          sender: 'bot',
          text: data.reply,
          chips: data.chips || [],
          time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        }
      ]);
    } catch (e) {
      setMessages(prev => [
        ...prev,
        {
          id: Date.now() + 1,
          sender: 'bot',
          text: lang === 'kn' ? 'ಸಂಪರ್ಕ ದೋಷವಾಗಿದೆ.' : 'Connection error. Please try again.',
          time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        }
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: '1100px', margin: '0 auto', padding: '1.5rem 1rem 5rem' }}>
      {/* Hidden File Input */}
      <input
        type="file"
        accept="image/*"
        capture="environment"
        ref={fileInputRef}
        style={{ display: 'none' }}
        onChange={handleFileUpload}
      />

      {/* Top Banner & Language + District Selector */}
      <div style={{
        display: 'flex', flexWrap: 'wrap', justifyContent: 'space-between',
        alignItems: 'center', gap: '1rem', marginBottom: '1.5rem',
        padding: '1rem 1.25rem', background: 'rgba(15, 30, 20, 0.7)',
        border: '1px solid rgba(74, 222, 128, 0.25)', borderRadius: '16px'
      }}>
        <div>
          <h1 style={{ fontSize: '1.4rem', fontWeight: 800, color: '#f0fdf4', margin: 0, display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span>🌾</span> {greetings[lang].title}
          </h1>
          <p style={{ margin: '4px 0 0', fontSize: '0.85rem', color: '#86efac' }}>
            {greetings[lang].sub}
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          {/* District selector */}
          <select
            value={district}
            onChange={e => setDistrict(e.target.value)}
            style={{
              background: '#092314', color: '#ecfdf5', border: '1px solid #16a34a',
              borderRadius: '8px', padding: '6px 10px', fontSize: '0.85rem', fontWeight: 600
            }}
          >
            {KARNATAKA_DISTRICTS.map(d => <option key={d} value={d}>{d}</option>)}
          </select>

          {/* Language selector */}
          <div style={{ display: 'flex', background: '#05180d', borderRadius: '8px', border: '1px solid #16a34a', padding: 2 }}>
            {[
              { code: 'kn', label: 'ಕನ್ನಡ' },
              { code: 'en', label: 'English' },
              { code: 'hi', label: 'हिंदी' }
            ].map(l => (
              <button
                key={l.code}
                onClick={() => setLang(l.code)}
                style={{
                  background: lang === l.code ? '#16a34a' : 'transparent',
                  color: lang === l.code ? '#ffffff' : '#9ca3af',
                  border: 'none', padding: '5px 10px', borderRadius: '6px',
                  cursor: 'pointer', fontSize: '0.8rem', fontWeight: 700
                }}
              >
                {l.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* 6 Feature Action Cards Grid */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))',
        gap: '0.85rem',
        marginBottom: '1.5rem'
      }}>
        {featureCards.map(c => (
          <button
            key={c.id}
            onClick={() => handleCardClick(c)}
            style={{
              background: 'linear-gradient(135deg, rgba(16, 44, 28, 0.85) 0%, rgba(8, 24, 15, 0.95) 100%)',
              border: '1px solid rgba(74, 222, 128, 0.25)',
              borderRadius: '14px',
              padding: '1.1rem 1rem',
              textAlign: 'center',
              cursor: 'pointer',
              transition: 'all 0.2s ease',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '6px',
              minHeight: '100px',
              boxShadow: '0 4px 16px rgba(0, 0, 0, 0.2)'
            }}
            onMouseEnter={e => { e.currentTarget.style.transform = 'translateY(-2px)'; e.currentTarget.style.borderColor = '#4ade80'; }}
            onMouseLeave={e => { e.currentTarget.style.transform = 'translateY(0)'; e.currentTarget.style.borderColor = 'rgba(74, 222, 128, 0.25)'; }}
          >
            <span style={{ fontSize: '1.9rem' }}>{c.icon}</span>
            <div style={{ fontWeight: 700, fontSize: '0.98rem', color: '#f0fdf4' }}>{c.title[lang]}</div>
            <div style={{ fontSize: '0.75rem', color: '#94a3b8', lineHeight: 1.25 }}>{c.desc[lang]}</div>
          </button>
        ))}
      </div>

      {/* Interactive Conversation Container */}
      <div style={{
        background: 'rgba(8, 20, 13, 0.85)',
        border: '1px solid rgba(74, 222, 128, 0.2)',
        borderRadius: '16px',
        padding: '1.25rem',
        minHeight: '380px',
        maxHeight: '550px',
        overflowY: 'auto',
        display: 'flex',
        flexDirection: 'column',
        gap: '12px',
        boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4)'
      }}>
        {messages.map((m, idx) => (
          <div
            key={m.id || idx}
            style={{
              alignSelf: m.sender === 'user' ? 'flex-end' : 'flex-start',
              maxWidth: '85%',
              background: m.sender === 'user' ? '#166534' : 'rgba(16, 44, 28, 0.95)',
              border: m.sender === 'user' ? '1px solid #22c55e' : '1px solid rgba(74, 222, 128, 0.3)',
              borderRadius: '12px',
              padding: '12px 14px',
              color: '#f8fafc',
              fontSize: '0.92rem',
              lineHeight: 1.5,
              position: 'relative'
            }}
          >
            {m.previewUrl && (
              <img
                src={m.previewUrl}
                alt="Uploaded crop/soil"
                style={{ width: '100%', maxHeight: '200px', objectFit: 'cover', borderRadius: '8px', marginBottom: '8px' }}
              />
            )}
            {renderMessageContent(m.text)}

            {m.sender === 'bot' && (
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '8px', borderTop: '1px solid rgba(255,255,255,0.08)', paddingTop: '6px' }}>
                <button
                  onClick={() => speakText(m.text, idx)}
                  style={{
                    background: 'none', border: 'none', color: '#86efac',
                    cursor: 'pointer', fontSize: '0.78rem', display: 'flex', alignItems: 'center', gap: '4px'
                  }}
                >
                  {speakingIdx === idx ? '⏹️ Stop' : '🔊 Listen / ಕೇಳಿ'}
                </button>
                <span style={{ fontSize: '0.7rem', color: '#94a3b8' }}>{m.time}</span>
              </div>
            )}

            {/* Context chips */}
            {m.chips && m.chips.length > 0 && (
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', marginTop: '10px' }}>
                {m.chips.map((chip, cIdx) => (
                  <button
                    key={cIdx}
                    onClick={() => handleSend(chip)}
                    style={{
                      background: 'rgba(34, 197, 94, 0.15)',
                      border: '1px solid #22c55e',
                      color: '#bbf7d0',
                      borderRadius: '14px',
                      padding: '3px 10px',
                      fontSize: '0.75rem',
                      cursor: 'pointer'
                    }}
                  >
                    ✨ {chip}
                  </button>
                ))}
              </div>
            )}
          </div>
        ))}

        {loading && (
          <div style={{ alignSelf: 'flex-start', background: 'rgba(16, 44, 28, 0.95)', padding: '10px 14px', borderRadius: '12px', color: '#86efac', fontSize: '0.85rem' }}>
            🌾 {lang === 'kn' ? 'ರೈತಣ್ಣ ಕೃಷಿ AI ವಿಶ್ಲೇಷಿಸುತ್ತಿದೆ...' : 'Analyzing agriculture data...'}
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Bar */}
      <div style={{
        marginTop: '1rem',
        display: 'flex',
        alignItems: 'center',
        gap: '8px',
        background: 'rgba(12, 28, 18, 0.95)',
        border: '1px solid rgba(74, 222, 128, 0.35)',
        borderRadius: '30px',
        padding: '6px 12px',
        boxShadow: '0 4px 20px rgba(0, 0, 0, 0.3)'
      }}>
        {/* Camera photo button */}
        <button
          onClick={() => { setActiveFeature('detect_disease'); fileInputRef.current?.click(); }}
          title="Upload Plant / Soil Photo"
          style={{ background: 'none', border: 'none', fontSize: '1.3rem', cursor: 'pointer', padding: '4px' }}
        >
          📷
        </button>

        {/* Soil Health Card icon button */}
        <button
          onClick={() => setShowSoilModal(true)}
          title="Soil Health Card Form"
          style={{ background: 'none', border: 'none', fontSize: '1.3rem', cursor: 'pointer', padding: '4px' }}
        >
          🧪
        </button>

        {/* Voice recording button */}
        <button
          onClick={toggleVoiceRecording}
          title="Voice Speech Input"
          style={{
            background: recording ? '#dc2626' : 'none',
            border: 'none',
            fontSize: '1.3rem',
            cursor: 'pointer',
            padding: '4px',
            borderRadius: '50%'
          }}
        >
          {recording ? '🔴' : '🎙️'}
        </button>

        {/* Text Input */}
        <input
          type="text"
          value={input}
          placeholder={greetings[lang].placeholder}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && handleSend()}
          style={{
            flex: 1,
            background: 'transparent',
            border: 'none',
            color: '#f8fafc',
            fontSize: '0.95rem',
            outline: 'none',
            padding: '8px 4px'
          }}
        />

        {/* Send Button */}
        <button
          onClick={() => handleSend()}
          disabled={loading || !input.trim()}
          style={{
            background: '#16a34a',
            color: '#ffffff',
            border: 'none',
            borderRadius: '50%',
            width: '36px',
            height: '36px',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontWeight: 'bold',
            opacity: loading || !input.trim() ? 0.5 : 1
          }}
        >
          ➤
        </button>
      </div>

      {/* Trilingual and Voice indicator matching blueprint */}
      <div style={{ textAlign: 'center', marginTop: '0.65rem', fontSize: '0.8rem', color: '#94a3b8', letterSpacing: '0.3px' }}>
        Kannada · English · Hindi · Voice
      </div>

      {/* Soil Health Card Modal */}
      {showSoilModal && (
        <div style={{
          position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
          background: 'rgba(0, 0, 0, 0.75)', display: 'flex', alignItems: 'center',
          justifyContent: 'center', zIndex: 1000, padding: '1rem'
        }}>
          <div style={{
            background: '#0d2817', border: '1px solid #22c55e', borderRadius: '16px',
            maxWidth: '450px', width: '100%', padding: '1.5rem', color: '#f0fdf4'
          }}>
            <h3 style={{ margin: '0 0 1rem', fontSize: '1.15rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
              🧪 {lang === 'kn' ? 'ಮಣ್ಣು ಪರೀಕ್ಷೆ ವರದಿ (Soil Health Card)' : 'Soil Health Card Values'}
            </h3>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', marginBottom: '1.25rem' }}>
              <div>
                <label style={{ fontSize: '0.8rem', color: '#86efac' }}>pH (4.0 - 9.5)</label>
                <input
                  type="number" step="0.1" value={soilForm.ph}
                  onChange={e => setSoilForm({ ...soilForm, ph: e.target.value })}
                  style={{ width: '100%', padding: '6px', background: '#05180d', border: '1px solid #16a34a', color: '#fff', borderRadius: '6px' }}
                />
              </div>
              <div>
                <label style={{ fontSize: '0.8rem', color: '#86efac' }}>Organic Carbon (%)</label>
                <input
                  type="number" step="0.05" value={soilForm.oc}
                  onChange={e => setSoilForm({ ...soilForm, oc: e.target.value })}
                  style={{ width: '100%', padding: '6px', background: '#05180d', border: '1px solid #16a34a', color: '#fff', borderRadius: '6px' }}
                />
              </div>
              <div>
                <label style={{ fontSize: '0.8rem', color: '#86efac' }}>Nitrogen N (kg/ha)</label>
                <input
                  type="number" value={soilForm.n}
                  onChange={e => setSoilForm({ ...soilForm, n: e.target.value })}
                  style={{ width: '100%', padding: '6px', background: '#05180d', border: '1px solid #16a34a', color: '#fff', borderRadius: '6px' }}
                />
              </div>
              <div>
                <label style={{ fontSize: '0.8rem', color: '#86efac' }}>Phosphorus P (kg/ha)</label>
                <input
                  type="number" value={soilForm.p}
                  onChange={e => setSoilForm({ ...soilForm, p: e.target.value })}
                  style={{ width: '100%', padding: '6px', background: '#05180d', border: '1px solid #16a34a', color: '#fff', borderRadius: '6px' }}
                />
              </div>
              <div style={{ gridColumn: 'span 2' }}>
                <label style={{ fontSize: '0.8rem', color: '#86efac' }}>Potassium K (kg/ha)</label>
                <input
                  type="number" value={soilForm.k}
                  onChange={e => setSoilForm({ ...soilForm, k: e.target.value })}
                  style={{ width: '100%', padding: '6px', background: '#05180d', border: '1px solid #16a34a', color: '#fff', borderRadius: '6px' }}
                />
              </div>
            </div>

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '8px' }}>
              <button
                onClick={() => setShowSoilModal(false)}
                style={{ background: 'transparent', border: '1px solid #6b7280', color: '#d1d5db', padding: '6px 14px', borderRadius: '6px', cursor: 'pointer' }}
              >
                Cancel
              </button>
              <button
                onClick={handleSoilSubmit}
                style={{ background: '#16a34a', border: 'none', color: '#ffffff', padding: '6px 16px', borderRadius: '6px', fontWeight: 'bold', cursor: 'pointer' }}
              >
                {lang === 'kn' ? 'ವರದಿ ವಿಶ್ಲೇಷಿಸಿ' : 'Analyze Soil'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
