"""Safety checks for release-data restoration; does not run research."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts.restore_release_data import checked_path, restore


class ReleaseRestoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.assets = self.root / 'assets'
        self.target = self.root / 'restored'
        self.assets.mkdir()
        self.data = b'example market data'
        raw = self.assets / 'raw.parquet'
        raw.write_bytes(self.data)
        prefix = 'bundle/'
        report = b'research report'
        entries = [{'path': 'result/report.md', 'bytes': len(report),
                    'sha256': hashlib.sha256(report).hexdigest()}]
        bundle = self.assets / 'results.zip'
        with zipfile.ZipFile(bundle, 'w') as archive:
            archive.writestr(prefix + 'result/report.md', report)
            archive.writestr(prefix + '打包清单.json', json.dumps({'files': entries}))
            archive.writestr(prefix + '文件清单.csv', 'path,bytes,sha256')
            archive.writestr(prefix + '阅读指南.md', 'guide')
        def asset(path, **extra):
            return dict(asset_name=path.name, bytes=path.stat().st_size,
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest(), **extra)
        self.manifest = {'source_commit': 'test', 'assets': [
            asset(raw, kind='file', relative_path='data/raw.parquet'),
            asset(bundle, kind='zip', strip_prefix=prefix)]}
        self.save_manifest()

    def save_manifest(self):
        (self.assets / 'release-manifest.json').write_text(json.dumps(self.manifest), encoding='utf-8')

    def test_restore_and_idempotence(self):
        self.assertEqual(restore(self.assets, self.target)['written_files'], 2)
        self.assertEqual((self.target / 'data/raw.parquet').read_bytes(), self.data)
        self.assertEqual(restore(self.assets, self.target)['written_files'], 0)

    def test_verify_only_does_not_create_target(self):
        self.assertEqual(restore(self.assets, self.target, True)['verified_files'], 2)
        self.assertFalse(self.target.exists())

    def test_source_document_can_be_excluded_from_restore(self):
        self.manifest['assets'][1]['skip_paths'] = ['result/report.md']
        self.save_manifest()
        result = restore(self.assets, self.target)
        self.assertEqual(result['written_files'], 1)
        self.assertFalse((self.target / 'result/report.md').exists())

    def test_corrupt_asset_prevents_all_writes(self):
        (self.assets / 'results.zip').write_bytes(b'corrupt')
        with self.assertRaises(ValueError):
            restore(self.assets, self.target)
        self.assertFalse(self.target.exists())

    def test_conflict_prevents_all_writes(self):
        destination = self.target / 'result/report.md'
        destination.parent.mkdir(parents=True)
        destination.write_text('existing user work')
        with self.assertRaises(FileExistsError):
            restore(self.assets, self.target)
        self.assertFalse((self.target / 'data/raw.parquet').exists())
        self.assertEqual(destination.read_text(), 'existing user work')

    def test_path_escape_and_git_are_rejected(self):
        for value in ['../outside', '/absolute', 'C:/absolute', 'a\\b', '.git/config']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                checked_path(self.target, value)

    def test_duplicate_destinations_are_rejected(self):
        self.manifest['assets'].append(self.manifest['assets'][0])
        self.save_manifest()
        with self.assertRaises(ValueError):
            restore(self.assets, self.target)
        self.assertFalse(self.target.exists())


if __name__ == '__main__':
    unittest.main()
