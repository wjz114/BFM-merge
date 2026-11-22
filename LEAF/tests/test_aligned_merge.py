"""Numerical checks for prediction preservation and distillation gradients."""
import unittest
import numpy as np
import torch
from aligned_support import distillation_loss
from cpsc_fold import contrast_basis, regularized_procrustes
from projection_repair import ProjectionDelta, plain_state, projection_features, prototype_union_basis
from expert_alignment import capture_hidden, expert_alignment_terms


class AlignmentTests(unittest.TestCase):
    def test_rotation_fold_preserves_probabilities(self):
        rng = np.random.default_rng(42)
        p = rng.normal(size=(4, 24))
        p /= np.linalg.norm(p, axis=1, keepdims=True)
        q, n, _ = contrast_basis(p)
        self.assertEqual(q.shape[1], 3)
        w, b, h = rng.normal(size=(24, 8)), rng.normal(size=24), rng.normal(size=(32, 8))
        z = h @ w.T + b
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        base = rng.normal(size=z.shape)
        rp, _, _ = regularized_procrustes(z @ n, base @ n, 1.)
        r = q @ q.T + n @ rp @ n.T
        np.testing.assert_allclose(r.T @ r, np.eye(24), atol=1e-12)
        np.testing.assert_allclose(r @ q, q, atol=1e-12)
        folded = h @ (r @ w).T + r @ b
        folded /= np.linalg.norm(folded, axis=1, keepdims=True)
        np.testing.assert_allclose(folded, z @ r.T, atol=1e-12)
        old, new = torch.tensor(z @ p.T), torch.tensor(folded @ p.T)
        torch.testing.assert_close(old.softmax(-1), new.softmax(-1), atol=1e-12, rtol=0)
        y = torch.arange(len(z)) % 4
        torch.testing.assert_close(torch.nn.functional.cross_entropy(old, y),
                                   torch.nn.functional.cross_entropy(new, y), atol=1e-12, rtol=0)
        # Only contrast directions are protected: a common logit offset is legal.
        self.assertGreater((old - new).abs().max().item(), 1e-4)

    def test_degenerate_complement_is_identity(self):
        for dim in (0, 8):
            r, mu, reason = regularized_procrustes(np.zeros((10, dim)), np.zeros((10, dim)), 1.)
            np.testing.assert_array_equal(r, np.eye(dim))
            self.assertEqual(mu, 0.)
            self.assertIsNotNone(reason)

    def test_kd_teacher_stopped_student_differentiable(self):
        for kind in ('semkd', 'logitkd'):
            torch.manual_seed(7)
            student = torch.randn(6, 12, requires_grad=True)
            teacher = torch.randn(6, 12, requires_grad=True)
            p = torch.nn.functional.normalize(torch.randn(3, 12), dim=-1)
            loss = distillation_loss(student, teacher, p, kind)
            loss.backward()
            self.assertIsNone(teacher.grad)
            self.assertTrue(torch.isfinite(student.grad).all())
            self.assertGreater(student.grad.norm().item(), 0.)
            self.assertAlmostEqual(distillation_loss(teacher, teacher, p, kind).item(), 0., places=6)

    def test_restricted_projection_update_and_plain_export(self):
        torch.manual_seed(11)
        original = torch.nn.Linear(8, 24).double()
        q = torch.linalg.qr(torch.randn(24, 3, dtype=torch.double))[0]
        projection = ProjectionDelta(original, q)
        x = torch.randn(9, 8, dtype=torch.double)
        torch.testing.assert_close(projection(x), original(x), atol=0, rtol=0)
        projection(x).square().mean().backward()
        self.assertGreater(projection.delta_weight.grad.norm().item(), 0.)
        with torch.no_grad():
            projection.delta_weight -= .1 * projection.delta_weight.grad
            projection.delta_bias -= .1 * projection.delta_bias.grad
        change = projection(x) - original(x)
        torch.testing.assert_close(change @ (torch.eye(24) - q @ q.T),
                                   torch.zeros_like(change), atol=1e-12, rtol=0)
        leaf = torch.nn.Module()
        leaf.qformer = torch.nn.Module()
        leaf.qformer.output_embed = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Identity(), projection)
        state = plain_state(leaf)
        self.assertEqual(set(state), {'qformer.output_embed.2.weight', 'qformer.output_embed.2.bias'})
        restored = torch.nn.Linear(8, 24).double()
        restored.load_state_dict({k.rsplit('.', 1)[-1]: v for k, v in state.items()}, strict=True)
        torch.testing.assert_close(restored(x), projection(x), atol=1e-12, rtol=0)

    def test_union_basis_does_not_add_between_task_centroids(self):
        p = torch.tensor([[1., 2., 0.], [-1., 2., 0.]], dtype=torch.double)
        q, _ = prototype_union_basis([p, p + torch.tensor([0., 0., 5.])])
        self.assertEqual(q.shape, (3, 1))
        torch.testing.assert_close(q @ q.T, torch.diag(torch.tensor([1., 0., 0.])), atol=1e-6, rtol=0)

    def test_frozen_feature_cache_keeps_order_and_training_rng(self):
        class TinyLeaf(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.qformer = torch.nn.Module()
                self.qformer.output_embed = torch.nn.Sequential(
                    torch.nn.Identity(), torch.nn.Identity(), torch.nn.Linear(8, 3))

            def forward(self, x, instruction):
                return self.qformer.output_embed(x), None

        leaf = TinyLeaf().eval()
        x, y = torch.randn(9, 8), torch.arange(9) % 3
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=4)
        state = torch.get_rng_state().clone()
        features, labels = projection_features(leaf, loader, torch.zeros(3), torch.device('cpu'))
        torch.testing.assert_close(features, x, atol=0, rtol=0)
        torch.testing.assert_close(labels, y, atol=0, rtol=0)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertFalse(features.requires_grad)

    def test_hidden_alignment_preserves_forward_and_student_gradients(self):
        torch.manual_seed(17)
        model = torch.nn.Sequential(torch.nn.Linear(6, 8), torch.nn.Tanh(), torch.nn.Linear(8, 4))
        x = torch.randn(5, 6)
        expected = model(x).detach()
        with capture_hidden(model, ['0', '2']) as hidden:
            student = model(x)
            torch.testing.assert_close(student, expected, atol=0, rtol=0)
            teacher = torch.randn_like(student, requires_grad=True)
            targets = {name: torch.randn_like(value, requires_grad=True) for name, value in hidden.items()}
            p = torch.nn.functional.normalize(torch.randn(3, 4), dim=-1)
            alpha = torch.full((2, 3), .6, requires_grad=True)
            kl, mse, anchor, _ = expert_alignment_terms(student, teacher, hidden, targets, p, alpha)
            (kl + mse + .1 * anchor).backward()
            self.assertGreater(model[0].weight.grad.norm().item(), 0.)
            self.assertGreater(model[2].weight.grad.norm().item(), 0.)
            self.assertTrue(torch.isfinite(alpha.grad).all())
            self.assertIsNone(teacher.grad)
            self.assertTrue(all(target.grad is None for target in targets.values()))
        self.assertTrue(all(not module._forward_hooks for module in model.modules()))

    def test_hidden_capture_cleanup_and_identical_alignment(self):
        model = torch.nn.Sequential(torch.nn.Linear(4, 4))
        with self.assertRaises(RuntimeError):
            with capture_hidden(model, ['0']):
                raise RuntimeError('synthetic interrupted forward')
        self.assertFalse(model[0]._forward_hooks)
        with capture_hidden(model, ['0']) as hidden:
            z = model(torch.randn(3, 4))
            p = torch.nn.functional.normalize(torch.randn(2, 4), dim=-1)
            kl, mse, anchor, _ = expert_alignment_terms(z, z, hidden, hidden, p, torch.full((2, 3), .5))
            self.assertAlmostEqual(kl.item(), 0., places=6)
            self.assertEqual(mse.item(), 0.)
            self.assertEqual(anchor.item(), 0.)


if __name__ == '__main__':
    unittest.main()
