import os
import tempfile
import unittest
from types import SimpleNamespace

from safetensors.torch import save_file as safe_save_file
import torch
import torch.nn as nn

from swift.dataset import RowPreprocessor
from swift.infer_engine import InferRequest
from swift.model.robot_state import load_robot_state_projector_state_dict, patch_robot_state_forward
from swift.template import Template


class DummyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.device = torch.device('cpu')
        self.config = SimpleNamespace(robot_state_token_id=99)
        self.embed_tokens = nn.Embedding(128, 4)
        self.robot_state_projector = nn.Linear(2, 4, bias=False)
        self.forward_kwargs = None

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, **kwargs):
        self.forward_kwargs = kwargs
        return inputs_embeds


class TestRobotState(unittest.TestCase):

    def test_cast_robot_states(self):
        row = {'robot_states': '[[1, 2], [3, 4]]'}
        RowPreprocessor._cast_robot_states(row)
        self.assertEqual(row['robot_states'], [[1.0, 2.0], [3.0, 4.0]])

        row = {'robot_states': [1, 2]}
        RowPreprocessor._cast_robot_states(row)
        self.assertEqual(row['robot_states'], [[1.0, 2.0]])

        row = {'robot_states': [[1, 2], [3]]}
        with self.assertRaises(ValueError):
            RowPreprocessor._cast_robot_states(row)

    def test_infer_request_robot_states(self):
        request = InferRequest(messages=[{'role': 'user', 'content': '<|robot_state|>'}], robot_states=[1.0, 2.0])
        self.assertEqual(request.robot_states, [[1.0, 2.0]])

    def test_data_collator_mm_data_robot_states(self):
        template = Template.__new__(Template)
        batch = [
            {'input_ids': [1, 2]},
            {'input_ids': [3], 'robot_states': torch.tensor([[1.0, 2.0]]), 'robot_state_count': 1},
            {'input_ids': [4], 'robot_states': torch.tensor([[3.0, 4.0], [5.0, 6.0]]), 'robot_state_count': 2},
        ]
        res = template._data_collator_mm_data(batch)
        self.assertEqual(res['robot_states'].shape, (3, 2))
        self.assertEqual(res['robot_state_counts'].tolist(), [0, 1, 2])

    def test_pre_forward_hook_passes_robot_state_to_model(self):
        template = Template.__new__(Template)
        template.mode = 'train'
        template.robot_state_token_id = 99
        model = DummyModel()
        inputs_embeds = torch.zeros(1, 3, 4)
        kwargs = {
            'input_ids': torch.tensor([[1, 99, 2]]),
            'inputs_embeds': inputs_embeds,
            'robot_states': torch.tensor([[3.0, 4.0]]),
        }
        _, new_kwargs = template.pre_forward_hook(model, None, kwargs)
        self.assertNotIn('input_ids', new_kwargs)
        self.assertIn('robot_states', new_kwargs)
        self.assertIn('_robot_state_input_ids', new_kwargs)

    def test_robot_state_forward_replaces_embeds(self):
        model = DummyModel()
        patch_robot_state_forward(model)
        with torch.no_grad():
            model.robot_state_projector.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]]))
        inputs_embeds = torch.zeros(1, 3, 4)
        output = model(
            inputs_embeds=inputs_embeds,
            _robot_state_input_ids=torch.tensor([[1, 99, 2]]),
            robot_states=torch.tensor([[3.0, 4.0]]))
        self.assertTrue(torch.allclose(output[0, 1], torch.tensor([3.0, 4.0, 7.0, 6.0])))
        self.assertNotIn('robot_states', model.forward_kwargs)

    def test_mixed_text_image_robot_state_embeds(self):
        model = DummyModel()
        patch_robot_state_forward(model)
        with torch.no_grad():
            model.robot_state_projector.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]]))
        inputs_embeds = torch.tensor([[
            [1.0, 1.0, 1.0, 1.0],
            [10.0, 20.0, 30.0, 40.0],
            [0.0, 0.0, 0.0, 0.0],
            [2.0, 2.0, 2.0, 2.0],
        ]])
        pixel_values = torch.ones(1, 3, 2, 2)
        image_grid_thw = torch.tensor([[1, 1, 1]])

        output = model(
            inputs_embeds=inputs_embeds,
            _robot_state_input_ids=torch.tensor([[11, 88, 99, 12]]),
            robot_states=torch.tensor([[3.0, 4.0]]),
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw)

        self.assertTrue(torch.allclose(output[0, 0], torch.tensor([1.0, 1.0, 1.0, 1.0])))
        self.assertTrue(torch.allclose(output[0, 1], torch.tensor([10.0, 20.0, 30.0, 40.0])))
        self.assertTrue(torch.allclose(output[0, 2], torch.tensor([3.0, 4.0, 7.0, 6.0])))
        self.assertTrue(torch.allclose(output[0, 3], torch.tensor([2.0, 2.0, 2.0, 2.0])))
        self.assertIs(model.forward_kwargs['pixel_values'], pixel_values)
        self.assertIs(model.forward_kwargs['image_grid_thw'], image_grid_thw)
        self.assertNotIn('robot_states', model.forward_kwargs)
        self.assertNotIn('robot_state_counts', model.forward_kwargs)
        self.assertNotIn('_robot_state_input_ids', model.forward_kwargs)

    def test_robot_state_forward_builds_embeds(self):
        model = DummyModel()
        patch_robot_state_forward(model)
        with torch.no_grad():
            model.robot_state_projector.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]]))
        output = model(input_ids=torch.tensor([[1, 99, 2]]), robot_states=torch.tensor([[3.0, 4.0]]))
        self.assertTrue(torch.allclose(output[0, 1], torch.tensor([3.0, 4.0, 7.0, 6.0])))

    def test_load_robot_state_projector_state_dict(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, 'model.safetensors')
            safe_save_file({'robot_state_projector.net.0.weight': torch.ones((4, 2))}, path)
            state_dict = load_robot_state_projector_state_dict(tmp_dir)
            self.assertIn('net.0.weight', state_dict)
            self.assertTrue(torch.equal(state_dict['net.0.weight'], torch.ones((4, 2))))


if __name__ == '__main__':
    unittest.main()
