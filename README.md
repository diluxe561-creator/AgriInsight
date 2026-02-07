# AgriInsight

# 🌾 AgriInsight – AI-Powered Platform for Smallholder Farmers in Nigeria

AgriInsight is a lightweight, AI-powered web platform that helps farmers in Nigeria predict crop yield using basic environmental data such as rainfall, temperature, and fertilizer usage.

It is designed to run on low-end systems and work offline, making it suitable for rural and developing regions.

---

## 🚀 Features

* ✅ AI-based crop yield prediction
* ✅ Simple web interface
* ✅ Offline support
* ✅ SQLite database storage
* ✅ Lightweight and fast
* ✅ Beginner-friendly codebase

---

## 🛠️ Tech Stack

| Component | Technology            |
| --------- | --------------------- |
| Backend   | Python (Flask)        |
| AI Model  | Scikit-learn          |
| Frontend  | HTML, CSS, JavaScript |
| Database  | SQLite                |
| Storage   | Joblib                |

---

## 📁 Project Structure

```
AgriInsight/
│
├── app.py
├── model.py
├── crop_model.pkl
├── farmers.db
│
├── templates/
│   └── index.html
│
└── static/
    └── style.css
```

---

## 📦 Installation

### 1. Clone the Project

```bash
git clone https://github.com/yourusername/AgriInsight.git
cd AgriInsight
```

### 2. Install Dependencies

```bash
pip install flask scikit-learn pandas joblib
```

### 3. Train the AI Model

```bash
python model.py
```

### 4. Run the Server

```bash
python app.py
```

### 5. Open in Browser

```
http://127.0.0.1:5000
```

---

## 📊 How It Works

1. User enters farming data
2. Data is sent to Flask backend
3. AI model predicts crop yield
4. Result is returned instantly
5. Data is stored in SQLite database
```

---

## 📈 Future Improvements

* 🌦️ Real-time weather integration
* 📱 Mobile-friendly app
* 📷 Crop disease detection
* 🔐 User authentication
* 📊 Analytics dashboard
* 📩 SMS alerts

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repo
2. Create a new branch
3. Make changes
4. Submit a pull request

---

## 📜 License

MIT License – Free to use and modify.

---

## 👨‍💻 Author

**Prince Destiny (Deluxe)**
Junior Developer & Python Automation Engineer
Focused on building real-world solutions for Africa.

---

⭐ If you like this project, don’t forget to star it on GitHub!
