from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

def calculate_bleu(reference, hypothesis):
    # Apply smoothing to handle zero n-grams gracefully
    smoothing = SmoothingFunction().method1
    return sentence_bleu([reference.split()], hypothesis.split(), weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)

def score_chunk(chunk, ground_truth):
    """
    Evaluates a single chunk against a ground truth answer
    using BLEU (2-gram) and ROUGE scores and returns their average.
    
    Parameters:
    - chunk (str): The text chunk to evaluate
    - ground_truth (str): The ground truth answer text
    
    Returns:
    - float: The average score of BLEU-2 and ROUGE-L, scaled between 0 and 1
    """
    if not chunk:
        raise ValueError("The chunk is empty.")
    
    # Calculate BLEU-2 score (unigram + bigram, weights = 0.5 each)
    bleu_score = calculate_bleu(ground_truth, chunk)

    # Initialize ROUGE scorer and calculate ROUGE-L score
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = rouge.score(ground_truth, chunk)
    rouge_l_score = rouge_scores['rougeL'].fmeasure  # Use F-measure as a balanced score

    # Final average score (scale: 0 to 1)
    final_score = (bleu_score + rouge_l_score) / 2

    return final_score


