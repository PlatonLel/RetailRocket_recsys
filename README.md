# E-Commerce Recommender System

A modern, hybrid recommender system for e-commerce, leveraging Bayesian Personalized Ranking (BPR) matrix factorization. Includes a fast training pipeline and a beautiful Streamlit app for interactive inference.

## Features
- **Bayesian Personalized Ranking (BPR)** with matrix factorization
- Incorporates item properties and side information
- Fast, vectorized evaluation metrics (Hit@k, MRR@k, NDCG@k, ROC AUC)
- Leave-one-out evaluation protocol
- User/item ID mapping and feature encoding
- Streamlit web app for interactive recommendations

## Best metrics
- **Hit@10:** 0.8444
- **MRR@10:** 0.7059
- **NDCG@10:** 0.7398
- **ROC AUC:** 0.9405

## Project Structure
```
.
├── app.py                # Streamlit app for inference
├── train.py              # Model training, evaluation, and checkpointing
├── model.py              # BPR matrix factorization model definition
├── dataset.py            # PyTorch Dataset for BPR
├── data_processing.py    # Data loading, preprocessing, and encoding
├── requirements.txt      # Python dependencies
├── models/               # Saved model checkpoints and encoders
│   ├── recommender_model.pt
│   ├── user_encoder.pkl
│   ├── item_encoder.pkl
└── └── model_params.pkl

```

## Setup Instructions
*Python version==3.11.9*
1. **Clone the repository**
   ```bash
   git clone https://github.com/PlatonLel/RetailRocket_recsys
   cd RetailRocket_recsys
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Download and place data**
   - Place `events.csv`, `item_properties_part1.csv`, and `item_properties_part2.csv` in the `data/` directory.

## Training the Model
Run the following command to train the BPR model and save the best checkpoint:
```bash
python train.py
```
- Model and encoders will be saved in the `models/` directory.
- You can adjust training parameters in `train.py` or by editing the `main()` function call at the bottom.

## Inference: Streamlit App
Launch the web app to get recommendations for any user in the training set:
```bash
streamlit run app.py
```
- Enter a User ID (as used in the training data) and select the number of recommendations.
- The app will display the top-K recommended item IDs.

## Credits
- Built with [PyTorch](https://pytorch.org/) and [Streamlit](https://streamlit.io/)
- Dataset: [RetailRocket](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)
- Developed by PlatonLel

## License
MIT License 