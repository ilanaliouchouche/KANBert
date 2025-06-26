import unittest
import torch
from model.utils import KANBertConfig
from model.utils.rope import RoPE


class TestRoPE(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:

        cls.config = KANBertConfig(
            vocabulary_size=30522,
            hidden_dim=768,
            max_sequence_len=512,
            n_layers=12,
            intermediate_dim=3072,
            num_attention_heads=12,
            periodicity=10000
        )
        cls.rope = RoPE(cls.config)

    def setUp(self) -> None:

        self.batch_size = 2
        self.seq_length = 10
        self.num_attention_heads = self.config.num_attention_heads
        self.attention_head_dim = (self.config.hidden_dim //
                                   self.config.num_attention_heads)
        self.q_tensor = torch.randn(self.batch_size,
                                    self.num_attention_heads,
                                    self.seq_length,
                                    self.attention_head_dim)
        self.k_tensor = torch.randn(self.batch_size,
                                    self.num_attention_heads,
                                    self.seq_length,
                                    self.attention_head_dim)
        self.device = "cpu"
        self.rope.to(self.device)
        self.q_tensor = self.q_tensor.to(self.device)
        self.k_tensor = self.k_tensor.to(self.device)

    def test_forward_shape(self) -> None:

        new_q, new_k = self.rope(self.q_tensor, self.k_tensor)
        expected_shape = (self.batch_size,
                          self.num_attention_heads,
                          self.seq_length,
                          self.attention_head_dim)
        self.assertEqual(new_q.shape,
                         expected_shape,
                         f"Expected shape {expected_shape}, "
                         f"but got {new_q.shape}")
        self.assertEqual(new_k.shape,
                         expected_shape,
                         f"Expected shape {expected_shape}, "
                         f"but got {new_k.shape}")

    def test_forward_type(self) -> None:

        new_q, new_k = self.rope(self.q_tensor, self.k_tensor)
        self.assertIsInstance(new_q,
                              torch.Tensor,
                              "Output should be a torch.Tensor, "
                              f"but got {type(new_q)}")
        self.assertIsInstance(new_k,
                              torch.Tensor,
                              "Output should be a torch.Tensor, "
                              f"but got {type(new_k)}")

    def test_device(self) -> None:

        new_q, new_k = self.rope(self.q_tensor, self.k_tensor)
        self.assertEqual(new_q.device.type,
                         self.device,
                         f"Expected device {self.device}, "
                         f"but got {new_q.device}")
        self.assertEqual(new_k.device.type,
                         self.device,
                         f"Expected device {self.device}, "
                         f"but got {new_k.device}")


if __name__ == "__main__":
    unittest.main()
