import os
import torch
from argparse import ArgumentParser
import pandas as pd
from prime.utils import (
    read_seq,
    scan_max_mutant,
    score_mutant,
    device_picker,
)
from prime.model import Config, ForMaskedLM
import gdown

script_path = os.path.dirname(os.path.realpath(__file__))
weight_path = os.path.join(script_path, 'checkpoints/prime_base.pt')


DEFAULT_WEIGHTS_URL = 'https://drive.google.com/file/d/15ciPzoc8Am3xLrL23SlnxbYfn39CJ7F_/view?usp=sharing'


@torch.no_grad()
def main():
    psr = ArgumentParser()
    psr.add_argument("-f", "--fasta", type=str, required=True)
    psr.add_argument("-m", "--mutant", type=str, required=True)
    psr.add_argument("-s", "--save", type=str, required=True)
    args = psr.parse_args()

    os.makedirs(os.path.dirname(weight_path), exist_ok=True)

    if not os.path.exists(weight_path):
        print('fetching checkpoint ...')
        gdown.download(
            url=DEFAULT_WEIGHTS_URL,
            output=weight_path,
            quiet=False,
            fuzzy=True,
        )

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    

    sequence = read_seq(args.fasta)
    df = pd.read_csv(args.mutant)

    device = device_picker()
    model = ForMaskedLM(Config())
    model.load_state_dict(torch.load(weight_path))
    model.eval()
    model = model.to(device)

    df, sequence, offset = scan_max_mutant(df=df, seq=sequence)

    sequence_ids = model.tokenize(sequence).to(device)
    attention_mask = torch.ones_like(sequence_ids).to(device)
    logits = model(input_ids=sequence_ids, attention_mask=attention_mask)[0]
    logits = torch.log_softmax(logits, dim=-1)
    mutants = df["mutant"]
    scores = []
    for mutant in mutants:
        score = score_mutant(
            mutant, sequence, logits=logits, vocab=model.VOCAB, offset=offset
        )
        scores.append(score)
    df["predict_score"] = scores
    df.to_csv(args.save, index=False)


if __name__ == "__main__":
    main()
