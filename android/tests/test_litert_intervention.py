import sys
import unittest
from pathlib import Path

import numpy as np
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


if __name__ == '__main__':
  unittest.main()
