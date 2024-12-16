from transformers import AutoTokenizer, AutoModel
import torch

class TextEmbedder:
    def __init__(self, model_path="/Users/jungseokoh/Desktop/Cursor_prac_2024/product_factory/KB-Albert"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()

    def embed_text(self, text):
        """텍스트에 대한 임베딩 생성"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings.cpu().numpy() 