from model.kanbert import KANBert
from model.utils import KANBertConfig
import unittest

from transformers import AutoTokenizer, AutoModel
import torch


class TestKANBert(unittest.TestCase):
    """
    Unit tests for the KANBert model.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """
        Set up the official BERT model, tokenizer, and the KANBert model.
        """

        cls.model_name = "bert-base-uncased"
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        cls.bert_model = AutoModel.from_pretrained(cls.model_name)
        cls.config = KANBertConfig(
            vocabulary_size=30522,
            hidden_dim=768,
            max_sequence_len=512,
            n_layers=12,
            intermediate_dim=3072,
            num_attention_heads=12,
            periodicity=10000
        )
        cls.kanbert_model = KANBert(cls.config)

    def setUp(self) -> None:
        """
        Set up the inputs for the KANBert model.
        """

        self.text = "This is a file for unit tests"
        self.inputs = self.tokenizer(self.text,
                                     return_tensors='pt',
                                     add_special_tokens=False)
        self.seq_len = self.inputs.input_ids.shape[1]
        self.device = "cpu"
        self.kanbert_model.to(self.device)
        self.inputs['input_ids'] = self.inputs['input_ids'].to(self.device)

    def test_forward_shape(self) -> None:
        """
        Test the shape of the output of the KANBert model.
        """

        kanbert_output = self.kanbert_model(self.inputs['input_ids'])
        bert_output = self.bert_model(self.inputs['input_ids'])[0]

        self.assertEqual(kanbert_output.shape,
                         bert_output.shape,
                         f"Expected shape {bert_output.shape}, "
                         "but got {kanbert_output.shape}")

    def test_forward_type(self) -> None:
        """
        Test the type of the output of the KANBert model.
        """

        kanbert_output = self.kanbert_model(self.inputs['input_ids'])
        self.assertIsInstance(kanbert_output,
                              torch.Tensor,
                              "Output should be a torch.Tensor, but got "
                              f"{type(kanbert_output)}")

    def test_device(self) -> None:
        """
        Test the device of the output of the KANBert model.
        """

        kanbert_output = self.kanbert_model(self.inputs['input_ids'])
        self.assertEqual(kanbert_output.device.type,
                         self.device,
                         f"Expected device {self.device}, but got "
                         f"{kanbert_output.device}")

    def test_embeddings_shape(self) -> None:
        """
        Test the shape of the output of the embeddings layer of the KANBert
        model.
        """

        token_emb = self.kanbert_model.embeddings(self.inputs['input_ids'])
        expected_shape = (self.inputs['input_ids'].shape[0],
                          self.seq_len,
                          self.config.hidden_dim)
        self.assertEqual(token_emb.size(),
                         expected_shape,
                         f"Expected shape {expected_shape}, "
                         f"but got {token_emb.size()}")

    def test_embeddings_type(self) -> None:
        """
        Test the type of the output of the embeddings layer of the KANBert
        model.
        """

        token_emb = self.kanbert_model.embeddings(self.inputs['input_ids'])
        self.assertIsInstance(token_emb,
                              torch.Tensor,
                              f"Output should be a torch.Tensor, "
                              f"but got {type(token_emb)}")

    def test_embeddings_device(self) -> None:
        """
        Test the device of the output of the embeddings layer of the KANBert
        model.
        """

        token_emb = self.kanbert_model.embeddings(self.inputs['input_ids'])
        self.assertEqual(token_emb.device.type,
                         self.device,
                         f"Expected device {self.device}, "
                         f"but got {token_emb.device}")


if __name__ == "__main__":
    unittest.main()
