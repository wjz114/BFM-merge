import unittest

import torch

from LEAF import QFormerLayer, TransformerLayer
from regmean_support import advance_layer, capture_layer_inputs, layer_statistics, merge_layer, solve_regmean


class RegMeanTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_weighted_solution_and_unobserved_columns(self):
        weights = [torch.tensor([[1., 3., 8.]]), torch.tensor([[5., 7., 2.]])]
        grams = [torch.diag(torch.tensor([3., 1., 0.])), torch.diag(torch.tensor([1., 3., 0.]))]
        merged, audit = solve_regmean(weights, grams, .95)
        torch.testing.assert_close(merged, torch.tensor([[2., 6., 5.]]))
        self.assertEqual(audit['inactive_columns'], 1)
        self.assertLess(audit['relative_solve_residual'], 1e-12)
        self.assertLess(audit['objective_merged'], audit['objective_average'])

    def test_fused_attention_statistics_preserve_forward(self):
        tower = TransformerLayer(16, 4, 32, 0.).eval()
        qformer = QFormerLayer(16, 16, 4, 2).eval()
        x, q, kv = torch.randn(3, 5, 16), torch.randn(3, 2, 16), torch.randn(3, 10, 16)
        mask = torch.triu(torch.full((5, 5), -torch.inf), diagonal=1)
        for layer, args, kwargs, count in [
                (tower, (x,), {'src_mask': mask}, 6),
                (qformer, (q, kv), {}, 10)]:
            with torch.no_grad():
                before = layer(*args, **kwargs)
                grams, audit = layer_statistics(layer, [(args, kwargs)], torch.device('cpu'))
                after = layer(*args, **kwargs)
            self.assertEqual(len(grams), count)
            self.assertLess(audit['attention_reconstruction_max_abs'], 1e-6)
            torch.testing.assert_close(before, after, rtol=0, atol=0)
            state = layer.state_dict()
            merged, diagnostics = merge_layer([state, state], [grams, grams], .95)
            for name in state:
                torch.testing.assert_close(merged[name], state[name], rtol=1e-5, atol=1e-6)
            self.assertTrue(all(d['relative_solve_residual'] < 1e-10 for d in diagnostics.values()))

    def test_layer_input_capture_and_sequential_propagation(self):
        layers = torch.nn.Sequential(TransformerLayer(16, 4, 32, 0.), TransformerLayer(16, 4, 32, 0.)).eval()
        x = torch.randn(3, 5, 16)
        with torch.no_grad(), capture_layer_inputs(layers, ['0', '1']) as captured:
            reference = layers(x)
        batches = [captured['0']]
        for layer in layers:
            batches = advance_layer(layer, batches, torch.device('cpu'))
        torch.testing.assert_close(batches[0][0][0], reference, rtol=0, atol=0)
        self.assertFalse(layers[0]._forward_pre_hooks)
        self.assertFalse(layers[1]._forward_pre_hooks)


if __name__ == '__main__':
    unittest.main()
