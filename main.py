import json
import torch
import hashlib
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


MAX_LENGTH = 512 # Max number of tokens model will accept as input

# Device selection 
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

MODEL_NAME = "deepset/deberta-v3-large-squad2"

def best_span_in_chunk(
    start_logits_1d, 
    end_logits_1d,
    max_answer_len=60, 
    top_k=20
    ):
    """
    Finds the highest-scoring valid answer span within a single chunk.

    Takes the top-k start and top-k end positions and searches all pairs
    for the best combination where end >= start and the span fits within
    max_answer_len tokens. Scores each pair by summing its start and end
    logits.
    """
    
    # Added as a safeguard for reusability
    seq_len = start_logits_1d.size(0) 
    k = min(top_k, seq_len)

    # Indipendant selection of top-k start and end positions
    s_top = torch.topk(start_logits_1d, k)
    e_top = torch.topk(end_logits_1d, k)

    best_score = float("-inf") # Initialise to negative infinity to ensure first valid span is always accepted
    best_start = 0
    best_end = 0

    # Convert to plain python lists to make indexing easier
    s_vals = s_top.values.tolist()
    s_idxs = s_top.indices.tolist()
    e_vals = e_top.values.tolist()
    e_idxs = e_top.indices.tolist()

    
    for start_score, start_position in zip(s_vals, s_idxs):
        for end_score, end_position in zip(e_vals, e_idxs):
            # Only consider spans where the end token appears after the start token.
            if end_position < start_position:
                continue
            # skip the answer if it's longer than the max allowed answer length
            if (end_position - start_position + 1) > max_answer_len:
                continue
            
            # Calculate the final score by summing the start and end logits
            score = start_score + end_score

            # If the current span has a higher score, update the best span found so far
            if score > best_score:
                best_score = score
                best_start = start_position
                best_end = end_position

    return best_score, best_start, best_end


def run_pipeline():
    print(f"Using device: {DEVICE}")

    # Load the tokeniser and model
    tokeniser = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()


    # Load data from JSON file 
    input_path = Path("input.json")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Chunk each unique document and create TF-IDF vectors 
    # Cached by MD5 hash so shared docs aren't reprocessed
    doc_chunks = {}
    doc_tfidf = {}
    doc_vectors = {}

    print("Building document chunks and TF-IDF indices")
    for item in tqdm(data, desc="Indexing Documents"):
        full_document = item["document"]
        doc_hash = hashlib.md5(full_document.encode('utf-8')).hexdigest()
        
        if doc_hash not in doc_chunks:
            # Tokenise document to get character to token offset mappings
            doc_tokens = tokeniser(
                full_document,
                add_special_tokens=False,
                return_offsets_mapping=True,
                return_attention_mask=False,
            )
            
            offsets = doc_tokens["offset_mapping"]
            chunks = []
            
            # Slide a window of 400 tokens with 100 overlap
            for i in range(0, len(offsets), 400 - 100):
                window = offsets[i:i+400]
                if not window:
                    break
                start_char = window[0][0] #first character of first token 
                end_char = window[-1][1] #last character of last token 
                chunks.append(full_document[start_char:end_char])
                
            doc_chunks[doc_hash] = chunks
            
            # Build TF-IDF index over the chunks
            vectoriser = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1,2),
                max_features=50000,
                sublinear_tf=True
            )
            tfidf_matrix = vectoriser.fit_transform(chunks)
            
            doc_tfidf[doc_hash] = vectoriser
            doc_vectors[doc_hash] = tfidf_matrix

    predictions = [] # Store final predictions

    print(f"Starting inference on {len(data)} questions")
    for item in tqdm(data, desc="Answering Questions", unit="q"):

        q_id = item["question_id"]
        question = item["question"]
        full_document = item["document"] 
        doc_hash = hashlib.md5(full_document.encode('utf-8')).hexdigest()

        # Retrieve document data from precomputed chunk vecs and TF-IDF indices
        vectoriser = doc_tfidf[doc_hash]
        tfidf_matrix = doc_vectors[doc_hash]
        chunks = doc_chunks[doc_hash]

        # vectorise the question and rank chunks by cosine similarity 
        q_vec = vectoriser.transform([question])
        sims = cosine_similarity(q_vec, tfidf_matrix).flatten()
        top_k_indices = sims.argsort()[-6:][::-1] # Top 6 chunks based on TF-IDF similarity if the document has fewer than 6 chunks, return all of them

        retrieved_chunks = [chunks[i] for i in top_k_indices]

        # Prepare input for the model by concatenating the question and the retrieved chunks
        # Note for write up: this could be done more efficiently by concatenating the tokeniser outputs directly but this is clearer to read 
        inputs = tokeniser(
            [question] * len(retrieved_chunks),
            retrieved_chunks,
            max_length=MAX_LENGTH,
            truncation="only_second", # truncate the document to MAX_LENGTH tokens, keeping the question, this is done because the model has a maximum input length of 512 tokens. Never truncate the question
            return_offsets_mapping=True,
            padding="max_length", # Uniform length for batching
            return_tensors="pt"  
        ).to(DEVICE)

        offset_mapping = inputs.pop("offset_mapping").cpu()

        # Used as a speed up 
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )

        # extract (6, 512) start logits and end logits
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        # Used to calculate which chunk's answer to trust the most
        null_scores = start_logits[:, 0] + end_logits[:, 0]

        # remove cls, pads, sep and question tokens
        sequence_ids_list = [inputs.sequence_ids(i) for i in range(len(inputs["input_ids"]))]
        mask = torch.tensor([
            [(1 if s == 1 else 0) for s in seq] 
            for seq in sequence_ids_list
        ], device=DEVICE)

        start_logits = start_logits.masked_fill(mask == 0, -10000.0)
        end_logits = end_logits.masked_fill(mask == 0, -10000.0)

        # Move everything to cpu 
        start_logits_cpu = start_logits.cpu()
        end_logits_cpu = end_logits.cpu()
        null_scores_cpu = null_scores.cpu()
        n_chunks = start_logits_cpu.size(0)

        # Same logic used in best_span_chunk, initialise with -inf 
        best_calibrated_score = float("-inf")
        best_chunk_idx = 0
        best_start = 0
        best_end = 0

        for c in range(n_chunks):
            # Find best valid span within the chunk
            score, s, e = best_span_in_chunk(
                start_logits_cpu[c],
                end_logits_cpu[c],
                max_answer_len=60,
                top_k=20,
            )
            # How much does the model prefer this span over the 'no answer' prediction?
            calibrated_score = score - null_scores_cpu[c].item() 
            if calibrated_score > best_calibrated_score:
                best_calibrated_score = calibrated_score
                best_chunk_idx = c
                best_start = s
                best_end = e

        # Map winning token positions back to char in the chunk text
        start_char = offset_mapping[best_chunk_idx][best_start][0].item()
        end_char   = offset_mapping[best_chunk_idx][best_end][1].item()
        answer = retrieved_chunks[best_chunk_idx][start_char:end_char].strip()

        predictions.append({
            "question_id": q_id,
            "answer": answer if answer else "No answer found",
        })

    # Save final predictions to predictions.json 
    with open("predictions.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)

if __name__ == "__main__":
    run_pipeline()