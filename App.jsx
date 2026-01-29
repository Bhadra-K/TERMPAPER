import { useState } from "react";
import axios from "axios";

function App() {
  const [text, setText] = useState("");
  const [translation, setTranslation] = useState("");
  const [sourceLang, setSourceLang] = useState("auto");
  const [targetLang, setTargetLang] = useState("en");

  const translateText = async () => {
    try {
      const response = await axios.post("http://127.0.0.1:8000/translate", {
        text,
        target_lang: targetLang
      });
      setTranslation(response.data.translation);
    } catch (error) {
      console.error("Translation Error:", error);
    }
  };

  return (
    <div style={{ padding: "2rem", fontFamily: "Arial" }}>
      <h1>Multi-language Translator</h1>
      <textarea
        rows="5"
        cols="50"
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Enter text..."
      />
      <br />
      <select onChange={(e) => setTargetLang(e.target.value)} value={targetLang}>
        <option value="en">English</option>
        <option value="hi">Hindi</option>
        <option value="fr">French</option>
        {/* Add more languages as needed */}
      </select>
      <button onClick={translateText}>Translate</button>
      <h2>Translation:</h2>
      <p>{translation}</p>
    </div>
  );
}

export default App;
