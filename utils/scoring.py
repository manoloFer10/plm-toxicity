import torch, math
import numpy as np
from utils.toxic_scorers.toxDL2 import load_ToxDL2_model, load_domain2vector, pdb_to_graph
from pathlib import Path
from torch_geometric.data import Data

# Los modelos de toxicidad sólo pueden ver 50 aa. OK -> agarremos todas las proteínas de longitud mayor a 50 aa y veamos si las catalogan bien.

class ToxDL2Scorer():
    def __init__(self):
        self.model = self._load_model()
        self.domain2vector = self._load_domain2vector()
        self.device = 'cuda'
        


    def get_temp_pdb_structures(self, sequence):
        #TODO
        raise NotImplementedError
    
    def get_protein_domains(self, sequence):
        #TODO
        raise NotImplementedError
    
    def score(self, x):
        '''
        Should output the probs for (tox, non_tox).

        Returns: a tuple: (tox_prob, non_tox_prob)
        '''

        #TODO: implement sequence scoring. If the sequence is larger than 50 aminoacids, score with a sliding window and then combine the results 
        # One way: take the max prob score on the sliding window. That's the tox_prob, and non_tox_prob = 1 - tox_prob.

        # sliding window and call _score for each. If AF2 pLDDT is bad, don't average those samples.

        raise NotImplementedError('ToxicScore.score()')
        return self.model(x) 
    

    def _score(self, sequence):

        assert len(sequence)<50, 'Sequence length of sub string should be less than 50 for ToxDL2 to work.'

        pdb_file = self.get_temp_pdb_structure(sequence)
        protein_domains = self.get_protein_domains(sequence)

        protein_feature = self.obtain_protein_feature(pdb_file, protein_domains)

        with torch.no_grad():
            protein_feature = protein_feature.to(self.device)
            prediction = self.model.forward(protein_feature)
            print(protein_feature.name + f"\tPrediction: {prediction.item()}")

        return prediction
    
    
    def get_domain_vector(self, protein_domains):
        domain2vector_model = self.domain2vector
        domain_embeddings = [domain2vector_model.wv[domain]
                            for domain in protein_domains if domain in domain2vector_model.wv]
        if domain_embeddings:
            return np.expand_dims(np.mean(domain_embeddings, axis=0), axis=0)
        else:
            return np.expand_dims(np.zeros(domain2vector_model.vector_size), axis=0)
        
    
    def obtain_protein_feature(self, pdb_data_path, protein_domains):
        # Create a Data object for the current protein
        protein_node_feat, protein_edge_index, protein_name, protein_sequence = pdb_to_graph(Path(pdb_data_path))
        protein_length = len(protein_sequence)
        domain_vector = self.get_domain_vector(protein_domains)
        
        # unknown tested protein label information
        y = -1
        data_item = Data(
            x=protein_node_feat,
            edge_index=protein_edge_index,
            name=protein_name,
            sequence=protein_sequence,
            length=protein_length,
            vector=domain_vector,
            y=torch.tensor(float(y), dtype=torch.float),
        )
        return data_item


    def _load_model():
        model = load_ToxDL2_model(Path('utils/toxic_scorers\checkpoints\ToxDL2_model.pth'))
        model.to('cuda')
        model.eval()

        return model
    
    def _load_domain2vector():
        model = load_domain2vector()
        return 



TOXIC_SCORER= ToxDL2Scorer() # if we make an ensemble, change this line.


def score_toxicity(sequences):
    global TOXIC_SCORER

    model_probs = [TOXIC_SCORER.score(seq) for seq in sequences]

    toxic_probs = torch.tensor([prob[0] for prob in model_probs])
    non_toxic_probs = torch.tensor([prob[1] for prob in model_probs])

    estimated_toxic_prob = torch.mean(toxic_probs)
    estimated_non_toxic_prob = torch.mean(non_toxic_probs)

    return estimated_toxic_prob, estimated_non_toxic_prob


def calculatePerplexity(sequence, model, tokenizer_fn):
    input_ids = torch.tensor(tokenizer_fn(sequence)).unsqueeze(0) 
    input_ids = input_ids.to(model.device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    return math.exp(loss)