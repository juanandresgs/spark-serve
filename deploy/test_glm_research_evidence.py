"""Regression gates for importing the concluded, deliberately inconclusive GLM window."""
from pathlib import Path
import unittest
from spark_serve.evidence import load, comparison_reasons
ROOT=Path(__file__).resolve().parents[1]
class GlmResearchEvidence(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.records=load(ROOT)
 def get(self,name):return self.records['glm-window-20260930-'+name+'-v1']
 def test_timeout_is_not_a_speed_or_correctness_verdict(self):
  r=self.get('tf-context-near-limit-retrieval-and-generation')
  self.assertIsNone(r['accounting']['completion_tokens'])
  self.assertIsNone(r['measurements'][0]['value'])
  self.assertEqual(r['qualification']['correctness']['status'],'unknown')
  self.assertEqual(r['qualification']['performance']['status'],'unknown')
  self.assertEqual(r['failures'][0]['category'],'timeout_before_first_token')
 def test_eager_retrieval_does_not_hide_truncation_or_unknown_cache(self):
  r=self.get('eager-context-near-limit-retrieval-and-generation')
  self.assertEqual(r['accounting']['prompt_tokens'],841804)
  self.assertEqual(r['accounting']['completion_tokens'],8192)
  self.assertEqual(r['qualification']['correctness']['status'],'failed')
  c=self.get('eager-context-cached-continuation')
  self.assertIsNone(c['accounting']['cached_tokens'])
  self.assertEqual(c['measurements'][0]['conditions']['cache'],'unknown')
 def test_policy_cohorts_retain_failure_and_comparison_guards(self):
  for arm in ['a1','fixed4','fixed7','a2']:
   for c in [1,4,8]:
    r=self.get(f'policy-{arm}-c{c}')
    self.assertEqual(r['accounting']['requests_attempted'],8)
    self.assertEqual(r['failures'][0]['count'],8)
    self.assertEqual(r['qualification']['correctness']['status'],'unknown')
    self.assertTrue(comparison_reasons(r,r['measurements'][0],r,r['measurements'][0]))
 def test_component_scope_and_reviewed_choices_are_preserved(self):
  r=self.get('tf-compact-tail-micro-harness')
  self.assertEqual(r['environment']['hardware']['nodes'],1)
  self.assertIsNone(r['identity']['model_revision'])
  self.assertIsNone(r['environment']['software']['rank_image_digests'])
  self.assertIn('full-model',r['qualification']['correctness']['scope'])
  decisions=[r for r in self.records.values() if r['kind']=='recommendation']
  self.assertEqual({d['id'] for d in decisions}, {
   'qwen-recommendation-20260930-v1',
   'glm-recommendation-20260930-v1',
   'qwen-recommendation-20261001-v1',
  })
  self.assertFalse(any(ref['id'].startswith('glm-window') for d in decisions for ref in d['evidence_runs']))
if __name__=='__main__':unittest.main()
