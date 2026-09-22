import unittest
import numpy as np
from scripts.screen_short_windows import catalog,diagonal_common_mask,bootstrap
from utils.opening_ic import ICConfig,feature_registry
from utils.directional_review import reviewed_registry

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
