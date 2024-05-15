import torch
from transformers import WhisperForConditionalGeneration

def mix_language_embeddings(model: WhisperForConditionalGeneration, tokenizer, languages=['zh', 'en'], target_language='zh', weights=None):
    target_id = tokenizer.convert_tokens_to_ids(f"<|{target_language}|>")
    new_embedding = torch.zeros(model.model.decoder.embed_tokens.weight[target_id].shape, dtype=model.model.decoder.embed_tokens.weight[target_id].dtype)
    if weights is None:
        weights = [1/len(languages)] * len(languages)
    with torch.no_grad():
        for language, weight in zip(languages, weights):
            language_id = tokenizer.convert_tokens_to_ids(f"<|{language}|>")
            new_embedding += model.model.decoder.embed_tokens.weight[language_id] * weight
        model.model.decoder.embed_tokens.weight[target_id] = new_embedding
    return model