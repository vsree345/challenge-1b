
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

tokenizer.save_pretrained("./models/flan-t5-base")
model.save_pretrained("./models/flan-t5-base")

print("✅ flan-t5-base model and tokenizer saved to ./models/flan-t5-base")