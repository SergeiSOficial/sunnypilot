import sys
import unittest
from pathlib import Path

import flatbuffers
import numpy as np
from ai_edge_litert import schema_py_generated as schema
from ai_edge_litert.interpreter import Interpreter, OpResolverType

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from intervene_litert_suffix import intervene
from android.tests.test_litert_half_boundaries import tiny_model


class LiteRTInterventionTest(unittest.TestCase):
  def test_intervention_changes_only_downstream_value(self):
    source = tiny_model()
    changed = intervene(source, 'dense')
    runner = Interpreter(model_content=changed, num_threads=1,
                         experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    runner.allocate_tensors()
    for detail in runner.get_input_details():
      value = [[-2.0]] if detail['name'] == 'fold3_diagnostic_override' else [[5.0, 7.0]]
      runner.set_tensor(detail['index'], np.array(value, np.float32))
    runner.invoke()
    np.testing.assert_array_equal(runner.get_tensor(runner.get_output_details()[0]['index']), [[0.0]])
    self.assertNotEqual(source, changed)

  def test_rejects_invalid_or_repeated_targets(self):
    source = tiny_model()
    for target in ('missing', 'input', 'output'):
      with self.subTest(target=target), self.assertRaises(ValueError):
        intervene(source, target)
    with self.assertRaises(ValueError):
      intervene(intervene(source, 'dense'), 'dense')

  def test_two_named_interventions(self):
    model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(tiny_model(), 0))
    graph = model.subgraphs[0]
    graph.tensors.append(schema.TensorT())
    graph.tensors[-1].name, graph.tensors[-1].shape = 'final', np.array([1, 1], np.int32)
    graph.tensors[-1].type = schema.TensorType.FLOAT32
    graph.tensors[-1].buffer = 0
    op = schema.OperatorT()
    op.opcodeIndex, op.inputs, op.outputs = 1, np.array([4], np.int32), np.array([5], np.int32)
    graph.operators.append(op)
    graph.outputs = np.array([5], np.int32)
    builder = flatbuffers.Builder(1024)
    builder.Finish(model.Pack(builder), file_identifier=b'TFL3')
    source = bytes(builder.Output())
    changed = intervene(intervene(source, 'dense', 'override_a'), 'output', 'override_b')
    runner = Interpreter(model_content=changed, num_threads=1,
                         experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    runner.allocate_tensors()
    values = {'input': [[1, 1]], 'override_a': [[-3]], 'override_b': [[2]]}
    for detail in runner.get_input_details():
      runner.set_tensor(detail['index'], np.array(values[detail['name']], np.float32))
    runner.invoke()
    np.testing.assert_array_equal(runner.get_tensor(runner.get_output_details()[0]['index']), [[2]])
    with self.assertRaises(ValueError):
      intervene(changed, 'dense', 'override_a')


if __name__ == '__main__':
  unittest.main()
