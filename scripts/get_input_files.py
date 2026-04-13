import argparse
import json
import os

from tqdm import tqdm

from utils.foldseek_util import get_struc_seq


def _load_questions(args: argparse.Namespace) -> list[str]:
    questions: list[str] = []
    if args.questions_file:
        with open(args.questions_file, encoding="utf-8") as qf:
            for line in qf:
                line = line.strip()
                if line:
                    questions.append(line)
    if args.question:
        questions.extend(args.question)
    if not questions:
        raise SystemExit("Provide at least one question via --question and/or --questions-file.")
    return questions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan structure files and write TSV rows (identifier, sequence, foldseek, question)."
    )
    parser.add_argument(
        "--foldseek",
        default="/zhouxibin/bin/foldseek",
        help="Path to the foldseek executable.",
    )
    parser.add_argument(
        "--rewrite",
        action="store_true",
        help="Overwrite result_input_file if it already exists.",
    )
    parser.add_argument(
        "--question",
        action="append",
        default=None,
        metavar="TEXT",
        help="Question string (repeat flag for multiple questions).",
    )
    parser.add_argument(
        "--questions-file",
        metavar="PATH",
        help="Text file with one question per line (empty lines skipped).",
    )
    parser.add_argument(
        "--structure-result",
        dest="structure_results",
        action="append",
        nargs=2,
        metavar=("STRUCTURE_PATH", "RESULT_INPUT_FILE"),
        required=True,
        help="Directory of PDB/CIF files and output TSV path. Repeat for multiple pairs.",
    )
    args = parser.parse_args()

    foldseek_bin = args.foldseek
    rewrite = args.rewrite
    question_list = _load_questions(args)

    for structure_path, result_input_file in args.structure_results:
        if not rewrite and os.path.exists(result_input_file):
            print(f"Write to {result_input_file} skipped, file exists.")
            continue
        with open(result_input_file, "w", encoding="utf-8") as f:
            for structure_file in tqdm(os.listdir(structure_path)):
                identifier = "_".join(structure_file.split(".")[:-1])
                if structure_file.endswith(".pdb") or structure_file.endswith(".cif"):
                    try:
                        result = get_struc_seq(
                            foldseek_bin,
                            os.path.join(structure_path, structure_file),
                            plddt_mask=True,
                        )
                        sequence, foldseek_seq, _sa_tokens = result["A"]
                        assert len(sequence) == len(foldseek_seq)
                        for question in question_list:
                            q = json.dumps(question, ensure_ascii=False)
                            f.write(f"{identifier}\t{sequence}\t{foldseek_seq.lower()}\t{q}\n")
                            f.flush()
                    except Exception as e:
                        print(e)
                        print(identifier)


if __name__ == "__main__":
    main()
