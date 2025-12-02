import cProfile
import pstats
from pstats import SortKey
from src.word_sketch import parse_corpus, search_in_parsed_corpus
import os, sys
from contextlib import redirect_stdout

cwd = os.path.dirname(__file__)

DATA_PATH = f"/home/wikror/gdrive/corpus-study/data/"

# query_name = "results-semantics-iteration2-2-separate-incl"
query_name = "results-communication-def-separate-incl"
JSON_PATH = f"{DATA_PATH}{query_name}.json"

def main():
    cProfile.run(
        f"parse_corpus(path_to_corpus=r'{JSON_PATH}', path_to_grammar='{cwd}/files/grammars/penn_3.1_en.txt', path_to_output='{DATA_PATH}parsed_corpora/{query_name}.p', KPWr=False, type='json_singlefile')",
        "restats")

    orig_stdout = sys.stdout

    with open(f"{DATA_PATH}ws_{query_name}_information.txt", "w") as f:
        with redirect_stdout(f):
            search_in_parsed_corpus(f"{DATA_PATH}parsed_corpora/{query_name}.p", "information", 20)

    # j = pstats.Stats('restats')
    # j.sort_stats(SortKey.CUMULATIVE).print_stats(15)


if __name__ == '__main__':
    main()