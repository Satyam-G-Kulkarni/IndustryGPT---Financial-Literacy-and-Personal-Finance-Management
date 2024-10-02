# IndustryGPT: Financial Literacy and Personal Finance Management Chatbot

## Problem Statement
In today's complex financial landscape, many individuals struggle to manage their personal finances due to a lack of accessible, reliable, and personalized financial education. This knowledge gap can result in poor financial decisions, leading to debt accumulation, inadequate savings, and insufficient retirement planning. Traditional financial advice is often generalized and fails to address the needs of individuals with limited financial literacy.

## Project Goal
IndustryGPT is an AI-powered chatbot designed to provide personalized financial advice and education. Leveraging Natural Language Processing (NLP), the chatbot delivers clear, accurate, and contextually relevant guidance on key personal finance topics, such as:
- Budgeting
- Saving
- Credit management
- Investment strategies
- Retirement planning

By democratizing access to financial knowledge, this project empowers users to make informed decisions and achieve better financial outcomes.

## Data Collection
The chatbot was trained using a collection of reputable financial education articles and standard datasets, ensuring the content was:
- Simplified
- Comprehensive
- Trustworthy

For extracting text from image-based PDF documents, Optical Character Recognition (OCR) was applied. A Python script using the `pytesseract` and `pdf2image` libraries was developed to convert PDFs into images and extract text, which was then saved for model training.

## Data Preprocessing
Data preprocessing involved merging datasets and creating a unified training set. Key steps included:
- Loading financial literacy datasets from Hugging Face and GitHub.
- Combining and transforming the datasets to fit a structured prompt-response format for model training.
- Performing a train-test split for proper evaluation.

## Model Training
Initially, GPT-2 was used for text generation, but its performance did not meet the project's specific requirements. Thus, the Llama-2-7B model was chosen for fine-tuning, using the QLoRA (Quantized Low-Rank Adaptation) technique to optimize training for memory efficiency.

### Key Training Parameters:
- **LoRA Configuration**: Rank = 64, Scaling = 16, Dropout = 0.1
- **Precision**: 4-bit precision with NF4 quantization
- **Training Setup**: 1 epoch, batch size of 4, learning rate = 2 × 10<sup>-4</sup>, using AdamW optimizer

## Inference and Web Interface
After fine-tuning, the Llama-2-7B model was integrated into a web interface using Flask to allow users to interact with the chatbot. The web app allows users to submit financial queries and receive personalized responses based on the fine-tuned model.

### Key Components:
- **HTML & CSS**: Provides a user-friendly chat interface where user and bot messages are displayed.
- **Flask Backend**: Handles user input, generates responses using the fine-tuned model, and returns JSON data.
- **Ngrok Integration**: Enables external access by tunneling the local server for public interaction.

## How to Run
To run the chatbot locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/IndustryGPT.git


