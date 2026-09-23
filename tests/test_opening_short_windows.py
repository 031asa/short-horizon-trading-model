import unittest
import numpy as np
from scripts.screen_short_windows import catalog,diagonal_common_mask,bootstrap
from utils.opening_ic import ICConfig,feature_registry
from utils.directional_review import reviewed_registry
from utils.short_window_analysis import pair_comparison,compression_supported

class ShortWindowTests(unittest.TestCase):
    def test_only_equal_window_versions_and_explicit_instant_baselines(self):
        r=reviewed_registry(feature_registry(ICConfig(history_seconds=tuple(range(1,11)))))
        groups,excluded=catalog(r)
        self.assertEqual(len(groups),28);self.assertEqual(len(excluded),4)
        self.assertEqual(sum(g['mode']=='rolling' for g in groups),21)
        for g in groups:
            if g['mode']=='rolling':
                for w,f in enumerate(g['factors'],1):self.assertTrue(f.endswith(f'_h{w}s'))
            else:self.assertEqual(len(set(g['factors'])),1)
        self.assertNotIn('F05_Momentum',{g['expression'] for g in groups})

    def test_common_uses_all_diagonal_versions_and_all_horizons_per_expression(self):
        x=np.ones((5,3,2));y=np.ones((5,3,30))
        x[1,2,0]=np.nan;y[3,0,29]=np.nan
        np.testing.assert_array_equal(diagonal_common_mask(x,y),[[False,False],[True,True],[False,True]])
        self.assertTrue(diagonal_common_mask(x,np.ones_like(y))[0].all())

    def test_day_bootstrap_constant_and_missing_days(self):
        values=np.array([.2,np.nan,.2,.2,.2]);draws=np.array([[0,1,2,3,4],[1,1,2,2,4]])
        np.testing.assert_allclose(bootstrap(values,draws),[.2,.2])
        self.assertTrue(np.isnan(bootstrap(np.full(5,np.nan),draws)).all())

    def test_paired_difference_excludes_unpaired_dates(self):
        short=np.array([[.2],[np.nan],[.4]])
        long=np.array([[.1],[.9],[np.nan]])
        result=pair_comparison(short,long,short,long,np.full_like(short,40),1,np.ones((10,3)))
        self.assertEqual(result['paired_days'][0],1)
        self.assertAlmostEqual(result['raw_difference'][0],.1)
        self.assertAlmostEqual(result['ci_low'][0],.1)

    def test_negative_prior_changes_difference_only_not_raw_ic(self):
        short=np.full((4,2),-.15);long=np.full((4,2),-.1)
        result=pair_comparison(short,long,short,long,np.full_like(short,50),-1,np.ones((10,4)))
        np.testing.assert_allclose(result['aligned_difference'],.05)
        np.testing.assert_allclose(result['raw_short_ic'],-.15)
        np.testing.assert_allclose(result['raw_difference'],-.05)

    def test_no_compression_success_when_both_windows_are_weak(self):
        rules=dict(ic_threshold=.05,minimum_valid_days=30,minimum_coverage=.8,minimum_agreeing_day_share=.6,compression_margin=.02)
        weak=dict(mean_rank_ic=.001,mean_ic=.001,valid_days=38,mean_coverage=.99,agreeing_day_share=.8)
        paired=dict(paired_days=38,coverage=.99,ci_low=0)
        self.assertFalse(compression_supported([weak]*3,[weak]*3,[paired]*3,1,rules))
        strong=dict(weak,mean_rank_ic=.1,mean_ic=.1)
        self.assertTrue(compression_supported([strong]*3,[strong]*3,[paired]*3,1,rules))
        loss=dict(paired,ci_low=-.03)
        self.assertFalse(compression_supported([strong]*3,[strong]*3,[paired,loss,paired],1,rules))
