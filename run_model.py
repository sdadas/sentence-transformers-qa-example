import logging
import sys
from dataclasses import dataclass, field
import faiss
from sentence_transformers import SentenceTransformer
from transformers import HfArgumentParser
from data_utils import DatasetReader


@dataclass
class QuestionAnsweringModuleArgs:
    model: str = field(
        default="./output",
        metadata={"help": "The name or path to the model"},
    )
    input_path: str = field(
        default="data.jsonl",
        metadata={"help": "Input data file"},
    )
    simple_qa: bool = field(
        default=False,
        metadata={"help": "Use question to question module"},
    )
    fp16: bool = field(
        default=True,
        metadata={"help": "Use 16 bit precision"},
    )


class Question2QuestionModule:

    def __init__(self, args: QuestionAnsweringModuleArgs):
        self.args = args
        self.model = SentenceTransformer(self.args.model)
        self.model.eval()
        if self.args.fp16: self.model.half()
        self.data = self._load_data()
        self.index = self._create_index()

    def _load_data(self):
        reader = DatasetReader(self.args.input_path, eval_size=0)
        return [pair for pair in reader.get_question_answer_pairs(first_answer_only=True)]

    def _create_index(self):
        logging.info("Initializing faiss index")
        texts = [q for q, a in self.data]
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        dim = embeddings.shape[1]
        res = faiss.IndexFlatIP(dim)
        res.add(embeddings)
        return res

    def generate_answer(self, question: str):
        q_emb = self.model.encode([question], normalize_embeddings=True, show_progress_bar=False)
        sim, indices = self.index.search(q_emb, 1)
        indices = indices.tolist()
        idx = indices[0][0]
        q, a = self.data[idx]
        return a


class Question2AnswerModule:

    def __init__(self, args: QuestionAnsweringModuleArgs):
        self.args = args
        self.model = SentenceTransformer(self.args.model)
        self.model.eval()
        if self.args.fp16: self.model.half()
        self.answers = self._load_answers()
        self.index = self._create_index()

    def _load_answers(self):
        reader = DatasetReader(self.args.input_path, eval_size=0)
        return [sent for sent in reader.get_answers()]

    def _create_index(self):
        logging.info("Initializing faiss index")
        embeddings = self.model.encode(self.answers, normalize_embeddings=True, show_progress_bar=True)
        dim = embeddings.shape[1]
        res = faiss.IndexFlatIP(dim)
        res.add(embeddings)
        return res

    def generate_answer(self, question: str):
        q_emb = self.model.encode([question], normalize_embeddings=True, show_progress_bar=False)
        sim, indices = self.index.search(q_emb, 1)
        indices = indices.tolist()
        idx = indices[0][0]
        return self.answers[idx]


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
    logging.root.setLevel(logging.DEBUG)
    parser = HfArgumentParser([QuestionAnsweringModuleArgs])
    args = parser.parse_args_into_dataclasses()[0]
    qa = Question2QuestionModule(args) if args.simple_qa else Question2AnswerModule(args)
    print("Dzień dobry, zapraszam do zadawania pytań")
    for question in sys.stdin:
        if question.strip() == "exit":
            print("Dziękuję za skorzystanie z usług wirtualnego asystenta")
            exit(0)
        answer = qa.generate_answer(question.strip())
        print(f"A: {answer}")
