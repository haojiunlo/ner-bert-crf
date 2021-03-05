import argparse
import json
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_jsonl",
        default=None,
        type=str,
        required=True,
        help="Path to input .jsonl",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        type=str,
        required=True,
        help="Path to output .txt",
    )
    args = parser.parse_args()


    data = []
    with open(args.input_jsonl) as r:
        for l in r:
            data.append(json.loads(l))

    with open(args.output_path, "w") as w:
        for d in tqdm(data):
            tags = ["O"] * len(d["text"])
            for ent in d["labels"]:
                tok_start = ent[0]
                tok_end = ent[1]
                ent_type = ent[-1]
                tags[tok_start : tok_end] = [f"B-{ent_type}"] + [f"I-{ent_type}"] * (tok_end - tok_start - 1)
            for token, tag in zip(d["text"], tags):
                w.write(f"{token}\t{tag}\n")
            w.write("\n")
