# 🖼️ Neural Style Transfer

This project implements Neural Style Transfer (NST), a technique that takes two images—a content image and a style image—and blends them together so that the output image looks like the content image but "painted" in the style of the other image.

## 🔧 Features
- Extracts content and style representations using a pre-trained CNN (typically VGG19)
- Combines content and style using loss minimization
- Option to run on GPU for faster performance
- Supports custom style/content images

## 🧠 Algorithm
Neural Style Transfer uses a loss function that combines:
- **Content Loss**: Difference between the content of the output image and the input content image.
- **Style Loss**: Difference between the style of the output image and the input style image.
- **Total Variation Loss** *(optional)*: Encourages spatial smoothness.

## 🚀 Setup

### 1. Clone the repo
```bash
git clone https://github.com/your-username/neural-style-transfer.git
cd neural-style-transfer
```

### 2. Install requirements
```bash
pip install -r requirements.txt
```

## 🖼️ Usage

```bash
python neural_style_transfer.py --content path/to/content.jpg --style path/to/style.jpg --output output.jpg
```

## 📁 File Structure
```
neural-style-transfer/
│
├── neural_style_transfer.py    # Main NST script
├── utils.py                    # Helper functions
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── images/                     # Example content/style images
└── outputs/                    # Generated output images
```

## 📚 References
- Gatys et al., *A Neural Algorithm of Artistic Style*, 2015
- PyTorch Tutorial: https://pytorch.org/tutorials/advanced/neural_style_tutorial.html

## 📝 License
MIT License