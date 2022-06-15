import json
import logging
import random
from typing import List, Dict, Tuple, Iterable


class DatasetReader:

    def __init__(self, input_path: str, eval_size: int=5000):
        self.input_path = input_path
        data: List[Dict] = self._load_data()
        if eval_size > 0:
            self.train = data[:-eval_size]
            self.eval = data[-eval_size:]
        else:
            self.train = data
            self.eval = []

    def _load_data(self) -> List[Dict]:
        logging.info(f"Loading data from {self.input_path}")
        results = []
        with open(self.input_path, "r", encoding="utf-8") as input_file:
            for line in input_file:
                json_value = json.loads(line.strip())
                results.append(json_value)
        logging.info(f"Loaded {len(results)} examples")
        random.seed(42)
        random.shuffle(results)
        return results

    def get_question_answer_pairs(self, split: str="train", first_answer_only: bool=False) -> Iterable[Tuple]:
        data = self.train if split == "train" else self.eval
        for item in data:
            for pair in self._get_question_answers(item, first_answer_only):
                yield pair

    def get_answers(self, split: str="train") -> Iterable[str]:
        data = self.train if split == "train" else self.eval
        for item in data:
            for pair in self._get_question_answers(item):
                yield pair[1]

    def _get_question_answers(self, item: Dict, first_answer_only: bool=False) -> Iterable[Tuple]:
        title = item.get("title").strip()
        if not title[-1] in ".?!": title += "?"
        text = item.get("text")
        if text is None: return
        question_text = " ".join([title, text.strip()])
        answers = item.get("answers")
        for answer in answers:
            answer_text = answer.get("text")
            if len(answer_text) < 5: continue
            yield (question_text, answer_text)
            if first_answer_only:
                return
