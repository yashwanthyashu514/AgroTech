import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  ScrollView,
  Image,
  ActivityIndicator,
  StyleSheet,
  Alert,
  SafeAreaView,
  StatusBar
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { Audio } from 'expo-av';

const API_BASE_URL = 'http://10.0.2.2:8000'; // Android emulator localhost alias

export default function ChatScreen() {
  const [lang, setLang] = useState('kn'); // 'kn' | 'en' | 'hi'
  const [messages, setMessages] = useState([
    {
      id: 'welcome',
      sender: 'bot',
      text: 'ನಮಸ್ಕಾರ! ನಿಮ್ಮ AI ಕೃಷಿ ಸಹಾಯಕ್ಕೆ ಸುಸ್ವಾಗತ. 🙏\nಸಸ್ಯ ಗುರುತಿಸಲು, ರೋಗ ಪತ್ತೆಹಚ್ಚಲು, ಮಣ್ಣಿನ ವರದಿ ವಿಶ್ಲೇಷಿಸಲು ಅಥವಾ ಪ್ರಶ್ನೆ ಕೇಳಲು ಕೆಳಗಿನ ಆಯ್ಕೆಗಳನ್ನು ಬಳಸಿ.',
      time: 'Just now',
    }
  ]);
  const [inputText, setInputText] = useState('');
  const [loading, setLoading] = useState(false);
  const [recording, setRecording] = useState(null);
  const [isRecording, setIsRecording] = useState(false);

  // 6 Primary Action Buttons matching blueprint
  const actions = [
    { id: 'plant', icon: '📷', title: lang === 'kn' ? 'ಸಸ್ಯ ಗುರುತಿಸಿ' : lang === 'hi' ? 'पौधे पहचानें' : 'Identify plant', type: 'camera' },
    { id: 'disease', icon: '🔍', title: lang === 'kn' ? 'ರೋಗ ಪತ್ತೆಹಚ್ಚಿ' : lang === 'hi' ? 'रोग का पता लगाएं' : 'Detect disease', type: 'camera' },
    { id: 'soil', icon: '🌱', title: lang === 'kn' ? 'ಮಣ್ಣು ವಿಶ್ಲೇಷಣೆ' : lang === 'hi' ? 'मिट्टी विश्लेषण' : 'Analyse soil', type: 'gallery' },
    { id: 'pest', icon: '🐛', title: lang === 'kn' ? 'ಕೀಟ ಗುರುತಿಸಿ' : lang === 'hi' ? 'कीट पहचानें' : 'Identify pests', type: 'camera' },
    { id: 'weather', icon: '🌦️', title: lang === 'kn' ? 'ಹವಾಮಾನ ಸಲಹೆ' : lang === 'hi' ? 'मौसम सलाह' : 'Weather advice', type: 'chat' },
    { id: 'ask_ai', icon: '💬', title: lang === 'kn' ? 'ಕೃಷಿ AI ಪ್ರಶ್ನಿಸಿ' : lang === 'hi' ? 'कृषि AI से पूछें' : 'Ask agriculture AI', type: 'chat' },
  ];

  const handleAction = async (action) => {
    if (action.type === 'camera') {
      const { status } = await ImagePicker.requestCameraPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('Permission needed', 'Camera permission is required to capture plant photos.');
        return;
      }
      const result = await ImagePicker.launchCameraAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        quality: 0.6, // Low-bandwidth optimization
      });
      if (!result.canceled && result.assets[0]) {
        uploadPhoto(result.assets[0].uri, action.id);
      }
    } else if (action.type === 'gallery') {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        quality: 0.6,
      });
      if (!result.canceled && result.assets[0]) {
        uploadPhoto(result.assets[0].uri, action.id);
      }
    } else {
      const prompt = lang === 'kn' 
        ? `${action.title} ಕುರಿತು ಕರ್ನಾಟಕದ ಪ್ರಸ್ತುತ ಕೃಷಿ ಶಿಫಾರಸು ತಿಳಿಸಿ.` 
        : `Provide agricultural advisory regarding ${action.title} for Karnataka.`;
      sendMessage(prompt);
    }
  };

  const uploadPhoto = async (uri, actionType) => {
    setLoading(true);
    const userMsg = {
      id: Date.now().toString(),
      sender: 'user',
      imageUri: uri,
      text: `[${actionType.toUpperCase()}] Photo uploaded`,
      time: 'Just now'
    };
    setMessages(prev => [...prev, userMsg]);

    const formData = new FormData();
    formData.append('file', {
      uri,
      name: 'photo.jpg',
      type: 'image/jpeg',
    });
    formData.append('lang', lang);
    formData.append('district', 'Davanagere');

    try {
      const res = await fetch(`${API_BASE_URL}/api/v1/plugin/raithanna/diagnose-image`, {
        method: 'POST',
        headers: { 'Content-Type': 'multipart/form-data' },
        body: formData,
      });
      const data = await res.json();
      setMessages(prev => [...prev, {
        id: (Date.now() + 1).toString(),
        sender: 'bot',
        text: data.reply || (lang === 'kn' ? data.response_kn : data.response_en) || 'Analysis complete.',
        time: 'Just now'
      }]);
    } catch (e) {
      setMessages(prev => [...prev, {
        id: (Date.now() + 1).toString(),
        sender: 'bot',
        text: 'Network error. Please ensure backend server is reachable.',
        time: 'Just now'
      }]);
    } finally {
      setLoading(false);
    }
  };

  const sendMessage = async (textToSend) => {
    const text = textToSend || inputText;
    if (!text.trim()) return;

    const userMsg = { id: Date.now().toString(), sender: 'user', text, time: 'Just now' };
    setMessages(prev => [...prev, userMsg]);
    setInputText('');
    setLoading(true);

    try {
      const res = await fetch(`${API_BASE_URL}/api/v1/plugin/raithanna/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, lang, district: 'Davanagere' }),
      });
      const data = await res.json();
      setMessages(prev => [...prev, {
        id: (Date.now() + 1).toString(),
        sender: 'bot',
        text: data.reply,
        time: 'Just now'
      }]);
    } catch (err) {
      setMessages(prev => [...prev, {
        id: (Date.now() + 1).toString(),
        sender: 'bot',
        text: 'Error connecting to server. Please try again.',
        time: 'Just now'
      }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#062e15" />
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.headerTitle}>🌾 AgroTech Karnataka</Text>
        <View style={styles.langSelector}>
          {['kn', 'en', 'hi'].map(l => (
            <TouchableOpacity key={l} onPress={() => setLang(l)} style={[styles.langBtn, lang === l && styles.langBtnActive]}>
              <Text style={[styles.langText, lang === l && styles.langTextActive]}>{l.toUpperCase()}</Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>

      {/* Quick Action Cards */}
      <View style={styles.actionGrid}>
        {actions.map(act => (
          <TouchableOpacity key={act.id} style={styles.actionCard} onPress={() => handleAction(act)}>
            <Text style={styles.actionIcon}>{act.icon}</Text>
            <Text style={styles.actionTitle}>{act.title}</Text>
          </TouchableOpacity>
        ))}
      </View>

      {/* Messages */}
      <ScrollView style={styles.messageList} contentContainerStyle={{ padding: 12 }}>
        {messages.map(msg => (
          <View key={msg.id} style={[styles.bubble, msg.sender === 'user' ? styles.userBubble : styles.botBubble]}>
            {msg.imageUri && <Image source={{ uri: msg.imageUri }} style={styles.previewImage} />}
            <Text style={[styles.messageText, msg.sender === 'user' ? styles.userText : styles.botText]}>{msg.text}</Text>
          </View>
        ))}
        {loading && <ActivityIndicator color="#10b981" style={{ marginVertical: 10 }} />}
      </ScrollView>

      {/* Input Bar */}
      <View style={styles.inputBar}>
        <TextInput
          style={styles.input}
          placeholder={lang === 'kn' ? "ಕೃಷಿ ಪ್ರಶ್ನೆ ಕೇಳಿ..." : "Ask your agriculture question..."}
          placeholderTextColor="#6b7280"
          value={inputText}
          onChangeText={setInputText}
        />
        <TouchableOpacity style={styles.sendButton} onPress={() => sendMessage()}>
          <Text style={styles.sendIcon}>➤</Text>
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#060c07' },
  header: { padding: 14, backgroundColor: '#0d381c', flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  headerTitle: { color: '#ffffff', fontSize: 17, fontWeight: '700' },
  langSelector: { flexDirection: 'row', gap: 6 },
  langBtn: { paddingVertical: 4, paddingHorizontal: 8, borderRadius: 12, backgroundColor: '#144c27' },
  langBtnActive: { backgroundColor: '#10b981' },
  langText: { color: '#9ca3af', fontSize: 11, fontWeight: '700' },
  langTextActive: { color: '#ffffff' },
  actionGrid: { flexDirection: 'row', flexWrap: 'wrap', padding: 8, justifyContent: 'space-between', backgroundColor: '#091c10' },
  actionCard: { width: '31%', backgroundColor: '#102e1c', paddingVertical: 10, paddingHorizontal: 6, borderRadius: 8, alignItems: 'center', marginBottom: 6 },
  actionIcon: { fontSize: 20, marginBottom: 4 },
  actionTitle: { color: '#d1fae5', fontSize: 11, fontWeight: '600', textAlign: 'center' },
  messageList: { flex: 1 },
  bubble: { maxWidth: '85%', padding: 12, borderRadius: 12, marginBottom: 8 },
  botBubble: { alignSelf: 'flex-start', backgroundColor: '#14301d', borderLeftWidth: 3, borderLeftColor: '#10b981' },
  userBubble: { alignSelf: 'flex-end', backgroundColor: '#047857' },
  messageText: { fontSize: 13, lineHeight: 18 },
  botText: { color: '#f3f4f6' },
  userText: { color: '#ffffff' },
  previewImage: { width: 180, height: 180, borderRadius: 8, marginBottom: 6 },
  inputBar: { flexDirection: 'row', padding: 10, backgroundColor: '#091c10', alignItems: 'center' },
  input: { flex: 1, height: 42, backgroundColor: '#132e1c', borderRadius: 21, paddingHorizontal: 16, color: '#ffffff', fontSize: 13 },
  sendButton: { width: 42, height: 42, borderRadius: 21, backgroundColor: '#10b981', marginLeft: 8, justifyContent: 'center', alignItems: 'center' },
  sendIcon: { color: '#ffffff', fontSize: 16, fontWeight: 'bold' },
});
