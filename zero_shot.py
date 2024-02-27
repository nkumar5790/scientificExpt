import torch
from transformers import BartForSequenceClassification, BartTokenizer
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler


class ZeroShotClassifier:
    """
    Initializes the object with the specified model name, labels, and batch size.

    Args:
        model_name (str): The name of the pre-trained model to be used.
        labels (list): The list of labels for sequence classification.
        batch_size (int): The batch size for processing inputs.

    Returns:
        None
    """
    def __init__(self, model_name="facebook/bart-large-mnli", labels=None,
                 batch_size=8):
        self.model = BartForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.labels = labels
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def zero_shot_classification_batch(self, texts):
        """
        Perform zero-shot classification on a batch of texts.

        Args:
            texts (List[str]): The list of texts to classify.

        Returns:
            List[List[Tuple[str, float]]]: A list of classification results for each text,
            where each result is a list of tuples containing the predicted label and the
            confidence score.
        """
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        dataset = TensorDataset(inputs["input_ids"], inputs["attention_mask"])
        dataloader = DataLoader(dataset, batch_size=self.batch_size, sampler=SequentialSampler(dataset))

        results = []
        with torch.no_grad():
            for batch in dataloader:
                batch_input_ids, batch_attention_mask = tuple(t.to(self.device) for t in batch)
                outputs = self.model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
                # Iterate over each element in the batch
                batch_results = []
                for probs, preds in zip(probabilities, logits.argmax(dim=1)):
                    batch_results.append([(self.labels[preds.item()], probs[preds].item())])
                results.extend(batch_results)
        return results


if __name__ == "__main__":
    # Example usage
    labels = ['entailment', 'contradiction', 'neutral']
    classifier = ZeroShotClassifier(labels=labels)

    texts = [
        "An apple a day keeps the doctor away.",
        "I am excited to go hiking this weekend.",
        "The Earth is flat.",
        "The quick brown fox jumps over the lazy dog."
    ]

    classification_results = classifier.zero_shot_classification_batch(texts)
    for text, results in zip(texts, classification_results):
        print(f"Text: {text}")
        print("Zero-shot classification results:")
        for label, probability in results:
            print(f"{label}: {probability:.4f}")
        print()
