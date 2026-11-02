import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from tools.soak_workstation import WorkstationCycle


class ExactModeTests(unittest.TestCase):
    def test_v1_or_truncated_exact_input_cannot_qualify_as_complete_exact(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for algorithm,leaves in [('reference-pixel-original-target-v1',424954),
                                     ('reference-pixel-exact-partition-v2',200000)]:
                with self.subTest(algorithm=algorithm,leaves=leaves):
                    source=root/(str(leaves)+algorithm+'.zip')
                    document=dict(plan=dict(planning_context=dict(revision=dict(inputs=dict(algorithms=dict(target=algorithm))))))
                    with zipfile.ZipFile(source,'x') as archive:
                        archive.writestr('project.json',json.dumps(document))
                    with patch('tools.soak_workstation.plan_values',return_value=dict(target_summary=dict(leaf_count=leaves))):
                        with self.assertRaisesRegex(ValueError,'complete paged exact'):
                            WorkstationCycle(source,root/'output',preserve_exact_plan=True)
                    self.assertFalse((root/'output').exists())
