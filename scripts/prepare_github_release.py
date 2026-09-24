"""Prepare checksummed GitHub attachments from existing outputs, without rerunning research."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'result/github_release'
REPOSITORY = '031asa/short-horizon-trading-model'
TAG = 'research-2026-09-24'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    packaged = json.loads((ROOT / 'result/研究结果打包验收.json').read_text(encoding='utf-8'))
    if packaged['source_commit'] != commit or not packaged['all_file_hashes_verified']:
        raise ValueError('Regenerate the result package at the current source commit first')
    archive = ROOT / 'result/研究结果汇总.zip'
    if sha(archive) != packaged['sha256']:
        raise ValueError('Research archive differs from its verified manifest')
    sources = [
        (archive, 'research-results.zip', dict(kind='zip', strip_prefix='短时交易模型研究结果/',
         skip_paths=['AGENTS.md', '项目交接.md', 'README.md', '撮合规则与研究说明.md', '项目口径核对.md'])),
        (ROOT / '新窗口交接_20260921/20260720_20260911_IC2609.parquet',
         '20260720_20260911_IC2609.parquet',
         dict(kind='file', relative_path='新窗口交接_20260921/20260720_20260911_IC2609.parquet')),
        (ROOT / 'result/opening_execution/daily_ic.parquet', 'daily_ic.parquet',
         dict(kind='file', relative_path='result/opening_execution/daily_ic.parquet')),
    ]
    assets = []
    for source, name, extra in sources:
        destination = OUT / name
        checksum = sha(source)
        if not destination.exists() or sha(destination) != checksum:
            shutil.copy2(source, destination)
        if sha(destination) != checksum:
            raise ValueError(f'Copy checksum mismatch: {name}')
        assets.append(dict(asset_name=name, source_path=source.relative_to(ROOT).as_posix(),
                           bytes=source.stat().st_size, sha256=checksum, **extra))
    shutil.copy2(ROOT / 'result/研究结果文件清单.csv', OUT / 'research-files.csv')
    manifest = dict(repository=REPOSITORY, private=True, tag=TAG, source_commit=commit,
                    created_at=datetime.now(timezone.utc).isoformat(), assets=assets,
                    result_file_count=packaged['files'],
                    excluded=['dependency and intermediate caches', 'cancelled date validation',
                              'duplicate summary_ic.csv (summary_ic.parquet retained)',
                              'earlier out-of-scope research and unrelated untracked files'],
                    restore_command='python -X utf8 scripts/restore_release_data.py --assets result/github_release/download')
    (OUT / 'release-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    names = [x['asset_name'] for x in assets] + ['release-manifest.json', 'research-files.csv']
    (OUT / 'SHA256SUMS.txt').write_text(''.join(f'{sha(OUT / name)}  {name}\n' for name in names), encoding='utf-8')
    print(json.dumps(dict(source_commit=commit, assets=assets, total_payload_bytes=sum(x['bytes'] for x in assets)), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
