"""Small arithmetic/accounting fixtures; no model, fit or saved result is opened."""
import json
from pathlib import Path
import unittest

import numpy as np

from scripts import review_stationary_direct_v70 as review

review.base.np=np


def partial():
    return dict(execution_status='incomplete',passed=False,verdict=dict(passed=False,complete=False),
        failures=[dict(stage='scientific_work')],work_seconds=1.,initial_identity_verified=True,completed_updates=dict(visual=0,constant=0),
        optimizer_step_entries=dict(visual=1,constant=0),pending_optimizer_arm='visual',
        optimizer_updates=0,optimizer_updates_upper_bound=1,optimizer_update_count_exact=False,
        model_forward_entries=dict(training=dict(visual=1,constant=0),evaluation=dict(visual=1,constant=0)),
        model_forward_returns=dict(training=dict(visual=1,constant=0),evaluation=dict(visual=1,constant=0)),
        model_forward_count_exact=True)


class DirectReviewFixtures(unittest.TestCase):
    def test_protocol_is_fixed_before_results(self):
        path=Path(__file__).resolve().parents[1]/review.REVIEW_PROTOCOL
        self.assertEqual(json.loads(path.read_text()),review.REVIEW_SPEC)
        self.assertEqual(review.REVIEW_SPEC['max_review_seconds'],120)

    def test_ap_groups_ties_and_empty_case_is_unknown(self):
        score=np.array([.8,.8,.1,.1],np.float32)
        target=np.array([True,False,True,False])
        self.assertEqual(review.base.average_precision_independent(score,target),.5)
        self.assertEqual(review.base.average_precision_independent(score[::-1],target[::-1]),.5)
        self.assertIsNone(review.base.average_precision_independent(score,np.zeros(4,bool)))

    def test_empty_bce_keeps_full_class_mass(self):
        target=np.array([[[True,False]],[[False,False]]])
        logits=np.array([[[0.,0.]],[[2.,2.]]],np.float32)
        loss,rows=review.base.bce_independent(logits,target)
        expected=(np.log(2.)+np.logaddexp(0.,2.))/2
        self.assertAlmostEqual(loss,expected,places=14)
        self.assertAlmostEqual(rows[1],np.logaddexp(0.,2.),places=14)

    def test_exact_component_sum_and_initial_zero(self):
        contextual=np.array([[[.2,-.3]]],np.float32)
        direct=np.zeros_like(contextual)
        review.check_components(contextual.copy(),contextual,direct,True)
        direct[0,0,0]=.1
        with self.assertRaises(ValueError): review.check_components(contextual.copy(),contextual,direct)
        with self.assertRaises(ValueError): review.check_components(contextual+direct,contextual,direct,True)

    def test_partial_mutation_bounds_and_returned_step_are_distinct(self):
        data=partial()
        self.assertEqual(review.check_accounting(data)['optimizer_upper_bound'],1)
        data['completed_updates']['visual']=1
        data['optimizer_updates']=data['optimizer_updates_upper_bound']=1
        data['pending_optimizer_arm']=None;data['optimizer_update_count_exact']=True
        self.assertIsNone(review.check_accounting(data)['pending_optimizer_arm'])
        data['optimizer_updates_upper_bound']=0
        with self.assertRaises(ValueError): review.check_accounting(data)

    def test_training_requires_verified_initial_identity_even_when_incomplete(self):
        data=partial();data['initial_identity_verified']=False
        with self.assertRaises(ValueError): review.check_accounting(data)

    def test_partial_journal_never_becomes_complete_or_continues_after_failure(self):
        data=partial()
        events=[dict(event='evaluation_started',arm='visual',checkpoint='initial',elapsed_seconds=.1),
                dict(event='evaluation_complete',arm='visual',checkpoint='initial',elapsed_seconds=.2),
                dict(event='update_started',arm='visual',update=1,elapsed_seconds=.3),
                dict(event='optimizer_step_enter',arm='visual',update=1,elapsed_seconds=.4),
                dict(event='failed',completed_updates=data['completed_updates'],pending_optimizer_arm='visual',elapsed_seconds=.5)]
        checked=review.incomplete_journal(events,data,17)
        self.assertEqual(checked['trailing_bytes'],17)
        events.append(dict(event='optimizer_step_return',arm='visual',update=1,elapsed_seconds=.6))
        with self.assertRaises(ValueError): review.incomplete_journal(events,data,0)

    def test_component_success_cannot_replace_sum_gate(self):
        def measured(value):
            return dict(mean_nonempty_iou=value,empty_anchor_false_positive_fractions=[0.,0.])
        values=dict(visual=measured(.1),constant=measured(0.),initial=measured(0.),zero=measured(0.),
                    descriptive_direct=measured(1.))
        self.assertFalse(review.base.verdict_independent(values)['passed'])


if __name__=='__main__':
    unittest.main()
