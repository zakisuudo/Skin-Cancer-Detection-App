import { useState } from "react";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    setFile(selected);

    if (selected) {
      setPreview(URL.createObjectURL(selected));
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    setLoading(true);
    setResult(null); // ✅ FIX 1

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      console.log("API RESPONSE:", data);

      setResult(data); // ✅ FIX 2 (IMPORTANT)
    } catch (err) {
      console.error(err);
      setResult({ label: "Error", explanation: "Server error", advice: "" });
    }

    setLoading(false);
  };

  const getClass = () => {
    if (!result) return "result";

    if (result.label === "Benign") return "result benign";
    if (result.label === "Malignant") return "result malignant";
    if (result.label === "No lesion") return "result neutral";

    return "result";
  };

  return (
    <div className="app">
      <div className="card">
        <h1>🧠 Skin AI Detector</h1>

        <label className="upload-btn">
          📁 Choose Image
          <input type="file" onChange={handleFileChange} hidden />
        </label>

        {preview && <img src={preview} alt="preview" className="preview" />}

        <div className="buttons">
          <button onClick={handleUpload}>
            Analyze
            <span className="ripple"></span>
          </button>

          <button onClick={() => window.location.reload()} className="reset">
            Reset
          </button>
        </div>

        {loading && (
          <div className="loader">
            <div className="spinner"></div>
            <p>🧠 AI is analyzing skin patterns...</p>
          </div>
        )}

        {result && (
          <div className="result-box">
            <div className={getClass()}>
              <h2>{result.label}</h2>
              <p>Confidence: {result.confidence}</p>
            </div>

            <div className="explain">
              <h4>🧠 AI Explanation</h4>
              <p>{result.explanation}</p>

              <h4>📌 Recommendation</h4>
              <p>{result.advice}</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;