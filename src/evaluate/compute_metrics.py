from rouge_score import rouge_scorer
import sacrebleu

def compute_metrics(predictions, references):
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=True
    )
    
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for key in rouge_scores:
            rouge_scores[key].append(scores[key].fmeasure)
    
    n = len(predictions)
    
    return {
        'BLEU': round(bleu.score, 2),
        'ROUGE-1': round(sum(rouge_scores['rouge1']) / n * 100, 2),
        'ROUGE-2': round(sum(rouge_scores['rouge2']) / n * 100, 2),
        'ROUGE-L': round(sum(rouge_scores['rougeL']) / n * 100, 2),
    }