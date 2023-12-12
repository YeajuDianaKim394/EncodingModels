"""Add embeddings to an events file that has words.

    salloc --mem=32G --time=00:15:00 --gres=gpu:1
    python code/embeddings.py -m gpt2-xl --layer 24
    python code/embeddings.py -m gptneo-3b --layer 16
    python code/embeddings.py -m llama2-7b --layer 16
    python code/embeddings.py -m llama2-7b-chat --layer 16
"""

from glob import glob

import pandas as pd
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from util.path import Path

# short names for long model names
EMBEDDINGS = {
    "llama2-7b": "meta-llama/Llama-2-7b-hf",
    "llama2-7b-chat": "meta-llama/Llama-2-7b-chat-hf",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
    "gptneo-3b": "EleutherAI/gpt-neo-2.7B",
}


def main(hfmodelname: str, device: str = "cpu", layer: int = 0):
    """Reimplemented to share model across iterations and for models with big context size."""

    dirname = f"model-{args.model}"
    if args.layer != -1:
        dirname += f"_layer-{args.layer}"

    # Find transcripts
    transpath = Path(
        root="stimuli", datatype="transcript", suffix="aligned", conv="*", ext=".csv"
    )
    search_str = transpath.starstr(["conv", "datatype"])
    files = glob(search_str)
    if not len(files):
        raise FileNotFoundError("No files found for: " + search_str)
    print(f"Found {len(files)} transcripts")

    tokenizer_args = dict()
    if "gpt2" in hfmodelname:
        tokenizer_args["add_prefix_space"] = True

    # Load model
    print("Loading model")
    tokenizer = AutoTokenizer.from_pretrained(hfmodelname, **tokenizer_args)
    model = AutoModelForCausalLM.from_pretrained(hfmodelname)
    model = model.eval()
    model = model.to(device)

    for tpath in tqdm(files):
        df = pd.read_csv(tpath)
        df.dropna(subset="word", inplace=True)

        # Tokenize input
        df.insert(0, "word_idx", df.index.values)
        df["hftoken"] = df.word.apply(tokenizer.tokenize)
        df = df.explode("hftoken", ignore_index=True)
        df["token_id"] = df.hftoken.apply(tokenizer.convert_tokens_to_ids)

        # Set up input
        tokenids = [tokenizer.bos_token_id] + df.token_id.tolist()
        batch = torch.tensor([tokenids], dtype=torch.long, device=device)

        # Run through model
        with torch.no_grad():
            output = model(batch, output_hidden_states=True)
            states = output.hidden_states[layer][0, 1:].numpy(force=True)

        df["embedding"] = [e for e in states]

        epath = Path.frompath(tpath)
        epath.update(root="features", datatype=dirname, suffix=None, ext="pkl")
        epath.mkdirs()
        df.to_pickle(epath)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-m", "--model", default="glove-50")
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()

    args.device = torch.device("cpu")
    if torch.cuda.is_available() and not args.force_cpu:
        args.device = torch.device("cuda", args.cuda)
    else:
        print("WARNING: using cpu only")

    args.hfmodelname = EMBEDDINGS.get(args.model, args.model)

    main(args.hfmodelname, device=args.device, layer=args.layer)
