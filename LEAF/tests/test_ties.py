import unittest

import torch

from ties_support import ties_vectors, transform_task_vectors


class TiesTests(unittest.TestCase):
    def test_reference_threshold_sign_and_disjoint_mean(self):
        x = torch.tensor([[1., -2., 3., -4., 5.], [-3., 2., -1., 4., 5.]])
        transformed, info = ties_vectors(x, .4)
        # Reference kth boundary retains >= the third magnitude in each row.
        # Column 3's exact sign tie follows the global positive majority.
        expected = torch.tensor([-3., 0., 3., 4., 5.])
        torch.testing.assert_close(transformed.mean(0), expected, rtol=0, atol=0)
        torch.testing.assert_close(torch.tensor(info['thresholds']), torch.tensor([3., 3.]))

    def test_all_zero_and_global_sign_tie(self):
        zeros, _ = ties_vectors(torch.zeros(2, 8), .2)
        self.assertEqual(zeros.count_nonzero(), 0)
        # The reference chooses negative entries when the global majority is zero.
        x = torch.tensor([[2., -2.], [-2., 2.]])
        result, _ = ties_vectors(x, .5)
        torch.testing.assert_close(result.mean(0), torch.tensor([-2., -2.]))

    def test_global_across_tensors_and_coefficient_gradient(self):
        deltas = {'a': torch.tensor([[1., 2., 3.], [-1., -2., 3.]]),
                  'b': torch.tensor([[4., -5.], [4., 5.]])}
        result, _ = transform_task_vectors(deltas, .4)
        direct, _ = ties_vectors(torch.cat(list(deltas.values()), dim=1), .4)
        torch.testing.assert_close(torch.cat(list(result.values()), dim=1), direct)
        coefficients = torch.full((2,), .5, requires_grad=True)
        output = sum((coefficients @ value).square().sum() for value in result.values())
        output.backward()
        self.assertTrue(torch.isfinite(coefficients.grad).all())
        self.assertTrue((coefficients.grad != 0).all())


if __name__ == '__main__':
    unittest.main()
