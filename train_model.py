import logging
import random
from dataclasses import dataclass, field
from typing import Any

from sentence_transformers import losses, SentenceTransformer, InputExample
from sentence_transformers.datasets import NoDuplicatesDataLoader
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from transformers import HfArgumentParser

from data_utils import DatasetReader


@dataclass
class SentenceTransformerTrainingArgs:
    model: str = field(
        default="paraphrase-multilingual-mpnet-base-v2",
        metadata={"help": "The name or path to the model"},
    )
    input_path: str = field(
        default="data.jsonl",
        metadata={"help": "Input data file"},
    )
    output_path: str = field(
        default="./output",
        metadata={"help": "Directory to store fine-tuned model"},
    )
    eval_steps: int = field(
        default=10_000,
        metadata={"help": "Number of steps after which the model is evaluated"},
    )
    fp16: bool = field(
        default=True,
        metadata={"help": "Use AMP for mixed precision training"},
    )
    batch_size: int = field(
        default=32,
        metadata={"help": "Batch size"},
    )
    lr: float = field(
        default=2e-6,
        metadata={"help": "Learning rate"},
    )
    num_train_epochs: int = field(
        default=3,
        metadata={"help": "Number of training epochs"},
    )
    warmup_ratio: float = field(
        default=0.1,
        metadata={"help": "Ratio of warmup steps"},
    )


class SentenceTransformerTrainer:

    def __init__(self, args: SentenceTransformerTrainingArgs):
        self.args = args
        self.base_model = SentenceTransformer(self.args.model)
        #self.base_model.max_seq_length = 512

    def train(self):
        loss = losses.MultipleNegativesRankingLoss(self.base_model)
        reader = DatasetReader(self.args.input_path)
        loader: Any = self._create_train_data_loader(reader)
        evaluator = self._create_evaluator(reader)
        warmup_steps = int(len(loader) * self.args.num_train_epochs * self.args.warmup_ratio)
        logging.info("Beginning training")
        self.base_model.fit(
            train_objectives=[(loader, loss)],
            epochs=self.args.num_train_epochs,
            warmup_steps=warmup_steps,
            output_path=self.args.output_path,
            show_progress_bar=True,
            use_amp=self.args.fp16,
            evaluation_steps=self.args.eval_steps,
            evaluator=evaluator,
            checkpoint_path=self.args.output_path,
            checkpoint_save_steps=self.args.eval_steps,
            checkpoint_save_total_limit=5,
            optimizer_params={'lr': self.args.lr}
        )

    def _create_train_data_loader(self, reader: DatasetReader):
        samples = [InputExample(texts=list(sample)) for sample in reader.get_question_answer_pairs()]
        return NoDuplicatesDataLoader(samples, self.args.batch_size)

    def _create_evaluator(self, reader: DatasetReader):
        sentences1, sentences2, labels = [], [], []
        for sample in reader.get_question_answer_pairs("eval"):
            sentences1.append(sample[0])
            sentences2.append(sample[1])
            labels.append(1)
        random_indices = list(range(len(sentences1)))
        random.shuffle(random_indices)
        for idx in range(1, len(sentences1)):
            sentences1.append(sentences1[random_indices[idx]])
            sentences2.append(sentences2[random_indices[idx-1]])
            labels.append(0)
        return BinaryClassificationEvaluator(sentences1, sentences2, labels, write_csv=False, name="eval")


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
    logging.root.setLevel(logging.DEBUG)
    parser = HfArgumentParser([SentenceTransformerTrainingArgs])
    args = parser.parse_args_into_dataclasses()[0]
    trainer = SentenceTransformerTrainer(args)
    trainer.train()
