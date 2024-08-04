from transformers import BartTokenizer, BartForConditionalGeneration

class Summarizer:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)

    def split_into_chunks(self, text, chunk_size=1024, overlap=256):
        words = text.split()
        for i in range(0, len(words), chunk_size - overlap):
            yield ' '.join(words[i:i + chunk_size])

    def summarize(self, text, chunk_size=1024, max_length=150, min_length=100, length_penalty=0.8, num_beams=4, repetition_penalty=2.0):
        # Adjust chunk size dynamically if the input is very large
        if len(text) > 5000:
            chunk_size = 512
        
        chunks = list(self.split_into_chunks(text, chunk_size))
        summaries = []
        
        for chunk in chunks:
            inputs = self.tokenizer(chunk, return_tensors="pt", truncation=True, padding='longest')
            if inputs.input_ids.size(1) > self.model.config.max_position_embeddings:
                print(f"Chunk size exceeds model's max input length: {inputs.input_ids.size(1)}")
                continue

            summary_ids = self.model.generate(
                inputs.input_ids,
                max_length=max_length,
                min_length=min_length,
                length_penalty=length_penalty,
                num_beams=num_beams,
                repetition_penalty=repetition_penalty,
                early_stopping=True
            )

            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            if summary:  # Check if summary is not empty
                summaries.append(summary)

        final_summary = self.refine_summary(summaries)
        return final_summary

    def refine_summary(self, summaries):
        return " ".join(summaries) if summaries else "No summary available."