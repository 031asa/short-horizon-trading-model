"""Load the project's Arrow runtime before pandas, then run the scoped tests."""
from pathlib import Path
import sys,json,hashlib,unittest
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'.cache/python-packages'))

if __name__=='__main__':
    out=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'result/opening_execution/.review_pending'
    result=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_opening*.py'))
    evidence=dict(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        passed=result.wasSuccessful(),checked_at=datetime.now(timezone.utc).isoformat(),
        test_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'tests').glob('test_opening*.py')})
    out.mkdir(parents=True,exist_ok=True)
    (out/'unit_test_verification.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    sys.exit(not result.wasSuccessful())
