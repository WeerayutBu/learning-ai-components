import contextlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from infrastructure.long_seq import process_long_input
from infrastructure import training


class LongInputTests(unittest.TestCase):
    def test_every_real_token_is_encoded_and_gradients_survive(self):
        weight = torch.tensor(1., requires_grad=True)
        def encoder(input_ids, attention_mask, output_attentions):
            n, width = input_ids.shape
            self.assertLessEqual(width, 512)
            return input_ids.float().unsqueeze(-1) * weight, [torch.ones(n, 1, width, width) / width]
        for length in [512, 1022, 1023, 1024]:
            ids = torch.arange(1, length + 1).unsqueeze(0).repeat(2, 1)
            mask = torch.ones_like(ids)
            mask[1, 400:] = 0
            output, attention = process_long_input(encoder, ids, mask, [101], [102])
            torch.testing.assert_close(output[0, :, 0], ids[0].float())
            torch.testing.assert_close(output[1, :400, 0], ids[1, :400].float())
            torch.testing.assert_close(attention[0, 0].sum(-1), torch.ones(length))
            if length > 1022:
                self.assertEqual(output[1, 400:].count_nonzero().item(), 0)
            output.sum().backward()
        self.assertGreater(weight.grad.item(), 0)


class TrainingLoopTests(unittest.TestCase):
    def run_loop(self, batches, accumulation):
        model = torch.nn.Linear(1, 1, bias=False)
        calls, gradients = [], []
        def forward(**kwargs):
            calls.append(1)
            return (model.weight.sum(),)
        model.forward = forward
        optimizer = Mock()
        optimizer.step.side_effect = lambda: gradients.append((len(calls), model.weight.grad.item()))
        args = SimpleNamespace(train_batch_size=4, gradient_accumulation_steps=accumulation,
                               warmup_ratio=0, max_grad_norm=0, evaluation_steps=-1,
                               save_path='test.pt', learning_rate=0.01, adam_epsilon=1e-6,
                               num_train_epochs=1, seed=66, n_gpu=0, device='cpu')
        batch = [torch.ones(1), torch.ones(1), [], [], []]
        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            loader = stack.enter_context(patch.object(training, 'DataLoader', return_value=[batch] * batches))
            stack.enter_context(patch.object(training, 'AdamW', return_value=optimizer))
            scheduler = stack.enter_context(patch.object(training, 'get_linear_schedule_with_warmup'))
            stack.enter_context(patch.object(training.amp, 'initialize', return_value=(model, optimizer)))
            stack.enter_context(patch.object(training.amp, 'scale_loss', side_effect=lambda loss, _: contextlib.nullcontext(loss)))
            evaluation = stack.enter_context(patch.object(training, 'evaluate', return_value=(1., {})))
            stack.enter_context(patch.object(training, 'report', return_value=[]))
            save = stack.enter_context(patch.object(training.torch, 'save'))
            stack.enter_context(patch('builtins.open', side_effect=lambda *a, **kw: Path(directory, 'result.json').open('w')))
            training.train(args, model, [1], [], [])
            self.assertFalse(loader.call_args.kwargs['drop_last'])
            self.assertEqual(scheduler.call_args.kwargs['num_training_steps'], len(gradients))
            self.assertEqual(scheduler.return_value.step.call_count, len(gradients))
            evaluation.assert_called_once()
            save.assert_called_once()
        return gradients

    def test_accumulation_and_partial_final_window(self):
        self.assertEqual(self.run_loop(6, 4), [(4, 1.), (6, 1.)])

    def test_one_batch_evaluates_and_saves(self):
        self.assertEqual(self.run_loop(1, 4), [(1, 1.)])
