# Handwritten Digit Recognition App

A web application that uses a Convolutional Neural Network (CNN) to recognize handwritten digits (0-9). Built with PyTorch and Streamlit.

![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white)

## 🎯 Features

- Upload images of handwritten digits
- Real-time AI prediction
- Confidence score for each prediction
- Probability distribution for all digits (0-9)
- Easy-to-use web interface

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **ML Framework**: PyTorch
- **Model**: CNN (Convolutional Neural Network)
- **Dataset**: MNIST

## 📁 Project Structure

```
Digit Recoginzer App/
├── app.py                      # Training script
├── streamlit_app.py            # Streamlit web app
├── requirements.txt            # Python dependencies
├── digit_recognition_model.pth # Trained PyTorch model
├── README.md                   # Project documentation
└── .gitignore                  # Git ignore file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/digit-recognizer.git
cd digit-recognizer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train the model (optional - model already included):
```bash
python app.py
```

4. Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

## 🌐 Deployment

### Deploy on Streamlit Cloud

1. Push your code to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub repository
4. Set the main file as `streamlit_app.py`
5. Deploy!

## 📊 Model Performance

- **Training Accuracy**: ~98%
- **Test Accuracy**: ~98%
- **Dataset**: MNIST (60,000 training images, 10,000 test images)

## 📝 License

MIT License

## 👤 Author

Your Name - [GitHub](https://github.com/yourusername)

## 🙏 Acknowledgments

- MNIST Dataset by Yann LeCun
- PyTorch Documentation
- Streamlit Documentation