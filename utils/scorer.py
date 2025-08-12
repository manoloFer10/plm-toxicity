import torch, math

# Los modelos de toxicidad sólo pueden ver 50 aa. OK -> agarremos todas las proteínas de longitud mayor a 50 aa y veamos si las catalogan bien.

class ToxicScorer():
    def __init__(self):
        self.model = ...
        #TODO
        
        
    
    def score(self, x):
        '''
        Should output the probs for (tox, non_tox).

        Returns: a tuple: (tox_prob, non_tox_prob)
        '''

        #TODO

        return self.model(x) 


TOXIC_SCORER= ToxicScorer()


def score_toxicity(sequences):
    global TOXIC_SCORER

    model_probs = [TOXIC_SCORER.score(seq) for seq in sequences]

    toxic_probs = torch.tensor([prob[0] for prob in model_probs])
    non_toxic_probs = torch.tensor([prob[1] for prob in model_probs])

    estimated_toxic_prob = torch.mean(toxic_probs)
    estimated_non_toxic_prob = torch.mean(non_toxic_probs)

    return toxic_probs, non_toxic_probs


def calculatePerplexity(sequence, model, tokenizer_fn):
    input_ids = torch.tensor(tokenizer_fn(sequence)).unsqueeze(0) 
    input_ids = input_ids.to(model.device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    return math.exp(loss)