import argparse
import random

from __init__ import generate

parser = argparse.ArgumentParser(
    description="Generate an image of a random Manchu word."
)
parser.add_argument("seed", nargs="?", help="seed (prompted for if omitted)")
parser.add_argument(
    "-r", "--random", action="store_true", help="use (and print) a random seed"
)
parser.add_argument(
    "-s", "--save", action="store_true", help="save the image instead of showing it"
)
parser.add_argument(
    "-o", "--output", help="path to save to with --save (default: <seed>.png)"
)
args = parser.parse_args()

if args.random:
    if args.seed is not None:
        parser.error("cannot pass a seed together with --random")
    seed = random.randrange(2**32)
    print(f"Seed: {seed}")
else:
    seed = args.seed if args.seed is not None else input("Seed: ")
    try:
        seed = int(seed)
    except ValueError:
        pass

image = generate(seed)
if args.save:
    output = args.output or f"{seed}.png"
    image.save(output)
    print(f"Saved {output}")
else:
    image.show()
