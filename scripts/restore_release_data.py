"""Verify and restore the private GitHub research snapshot (standard library only)."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import zipfile


def digest(stream):
    return hashlib.file_digest(stream, 'sha256').hexdigest()


def checked_path(root, relative):
    rel = PurePosixPath(relative)
    if (not relative or rel.is_absolute() or '\\' in relative or ':' in relative
            or any(p in {'.', '..', '.git'} for p in relative.split('/'))):
        raise ValueError(f'Unsafe restore path: {relative}')
    destination = (root / Path(*rel.parts)).resolve()
    if not destination.is_relative_to(root.resolve()):
        raise ValueError(f'Path escapes target: {relative}')
    return destination


def plan_restore(assets, target):
    manifest = json.loads((assets / 'release-manifest.json').read_text(encoding='utf-8'))
    plan = []
    for item in manifest['assets']:
        source = checked_path(assets, item['asset_name'])
        with source.open('rb') as stream:
            if source.stat().st_size != item['bytes'] or digest(stream) != item['sha256']:
                raise ValueError(f'Attachment checksum mismatch: {source.name}')
        if item['kind'] == 'file':
            plan.append((source, None, checked_path(target, item['relative_path']), item['sha256']))
        elif item['kind'] == 'zip':
            prefix = item['strip_prefix']
            with zipfile.ZipFile(source) as archive:
                names = archive.namelist()
                if len(names) != len(set(names)):
                    raise ValueError('Duplicate ZIP members')
                internal = json.loads(archive.read(prefix + '打包清单.json'))
                listed = {prefix + f['path'] for f in internal['files']}
                expected = listed | {prefix + n for n in ['打包清单.json', '阅读指南.md', '文件清单.csv']}
                if set(names) != expected:
                    raise ValueError('ZIP members differ from the internal manifest')
                for entry in internal['files']:
                    name = prefix + entry['path']
                    dest = checked_path(target, entry['path'])
                    with archive.open(name) as stream:
                        if (archive.getinfo(name).file_size != entry['bytes']
                                or digest(stream) != entry['sha256']):
                            raise ValueError(f'ZIP checksum mismatch: {name}')
                    # Source documents are supplied by Git; avoid replacing them
                    # with archive copies (including Windows line-ending changes).
                    if entry['path'] not in item.get('skip_paths', []):
                        plan.append((source, name, dest, entry['sha256']))
        else:
            raise ValueError(f'Unknown asset kind: {item["kind"]}')
    destinations = [p[2] for p in plan]
    if len(destinations) != len(set(destinations)):
        raise ValueError('Duplicate restore destinations')
    # Check every conflict before creating or writing any destination.
    for _, _, destination, checksum in plan:
        if destination.exists():
            with destination.open('rb') as stream:
                if digest(stream) != checksum:
                    raise FileExistsError(f'Different existing file; refusing to overwrite: {destination}')
    return manifest, plan


def restore(assets, target, verify_only=False):
    manifest, plan = plan_restore(assets.resolve(), target.resolve())
    written = 0
    if not verify_only:
        for source, member, destination, checksum in plan:
            if destination.exists():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            if member is None:
                with source.open('rb') as src, destination.open('xb') as dst:
                    shutil.copyfileobj(src, dst)
            else:
                with zipfile.ZipFile(source) as archive:
                    with archive.open(member) as src, destination.open('xb') as dst:
                        shutil.copyfileobj(src, dst)
            with destination.open('rb') as stream:
                if digest(stream) != checksum:
                    raise ValueError(f'Restored checksum mismatch: {destination}')
            written += 1
    return {'source_commit': manifest['source_commit'], 'verified_files': len(plan),
            'written_files': written, 'verify_only': verify_only, 'target': str(target.resolve())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', required=True, type=Path)
    parser.add_argument('--target', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(restore(args.assets, args.target, args.verify_only), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
