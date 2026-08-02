# X-MAP: Explainable Multi-Agent Pipeline for Cross-Modal Fake News Detection

X-MAP is a multimodal fake news detection project that combines textual and visual information for fake news classification using deep learning and explainable AI techniques.

## Features

- DistilBERT for text encoding
- ResNet-50 for image feature extraction
- Multimodal feature fusion
- LIME and Grad-CAM based explainability
- Evaluation on the Fakeddit dataset

## Repository Contents

- `X_MAP_improved (1).ipynb` – Main notebook
- `test_public.csv` – Dataset used for experimentation

## Technologies

- Python
- PyTorch
- Hugging Face Transformers
- Scikit-learn
- Pandas
- NumPy
- LIME
- Grad-CAM

## Results

| Metric | Score |
|--------|-------|
| Accuracy | 86.40% |
| Precision | 80.47% |
| Recall | 86.82% |
| F1 Score | 83.52% |

## Future Work

- Add training and inference scripts
- Release pretrained model
- Develop a web interface
- Extend evaluation on larger datasets

## License

This project is intended for academic and research purposes.
