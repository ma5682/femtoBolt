"""Inspect the local history database or make a consistent metadata backup."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from femto.history_store import HistoryStore
from femto.execution.history import ExecutionStore
from femto.io_utils import sha256_file, write_json_strict


def backup_metadata(outputs, destination):
    """Two online snapshots; interval is explicit, never claimed atomic across DBs."""
    destination = Path(destination).resolve()
    legacy_file = destination.suffix.lower() in ('.sqlite3', '.sqlite', '.db')
    execution_target = destination.with_name(destination.stem+'_execution.sqlite3') if legacy_file else destination/'execution.sqlite3'
    history_target = destination if legacy_file else destination/'history.sqlite3'
    manifest = destination.with_name(destination.stem+'_manifest.json') if legacy_file else destination/'manifest.json'
    targets = (history_target, execution_target, manifest)
    if any(path.exists() for path in targets) or not legacy_file and destination.exists():
        raise FileExistsError('Backup target already exists')
    history = HistoryStore(outputs)
    execution = ExecutionStore(outputs)
    if not history.path.exists() and not execution.path.exists():
        raise FileNotFoundError('No metadata databases exist')
    started = time.time()
    records = []
    for name, store, target in (('history.sqlite3', history, history_target),
                                ('execution.sqlite3', execution, execution_target)):
        if store.path.exists():
            begin = time.time()
            store.backup(target)
            records.append(dict(database=name, snapshot_file=target.name, sha256=sha256_file(target),
                                start_utc=begin, end_utc=time.time(), present=True))
        else:
            records.append(dict(database=name, snapshot_file=None, present=False))
    report = dict(schema='femto-metadata-backup-v1', databases=records, start_utc=started, end_utc=time.time(),
                  cross_database_atomic=False, possible_alignment_gap='changes during the recorded backup interval',
                  recovery='restore both snapshots without old WAL/SHM; incomplete runs stay interrupted/read-only',
                  includes_geometry_assets=False)
    write_json_strict(manifest, report)
    return dict(manifest=str(manifest), **report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outputs', required=True, type=Path)
    parser.add_argument('--backup', type=Path, help='New bundle directory (or legacy filename plus execution sibling and manifest)')
    args = parser.parse_args()
    store = HistoryStore(args.outputs)
    report = dict(database=str(store.path), present=store.path.exists())
    if store.path.exists():
      with store.connection() as connection:
        report.update(
                      integrity=[row[0] for row in connection.execute('PRAGMA integrity_check')],
                      records=connection.execute('SELECT count(*) FROM records').fetchone()[0],
                      hidden=connection.execute('SELECT count(*) FROM hidden').fetchone()[0],
                      collections=[dict(row) for row in connection.execute(
                          'SELECT collection,count(*) AS count FROM membership GROUP BY collection')])
    if report.get('integrity', ['ok']) != ['ok']:
        raise RuntimeError('History metadata integrity check failed: '+str(report['integrity']))
    if args.backup:
        report['backup'] = backup_metadata(args.outputs, args.backup)
    execution = ExecutionStore(args.outputs)
    report['execution'] = dict(database=str(execution.path), present=execution.path.exists(), runs=len(execution.runs()))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
