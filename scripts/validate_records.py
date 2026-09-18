#!/usr/bin/env python3
import json, sys
from pathlib import Path
root=Path(sys.argv[1]).resolve()
def load(path):
    with path.open(encoding='utf-8') as handle: return json.load(handle)
for path in sorted((root/'spec/schemas').glob('*.json')):
    schema=load(path)
    if schema.get('$schema')!='https://json-schema.org/draft/2020-12/schema': raise SystemExit(f'{path}: unexpected schema dialect')
    if schema.get('type')!='object' or not isinstance(schema.get('required'),list) or not isinstance(schema.get('properties'),dict): raise SystemExit(f'{path}: incomplete object schema')
records=[]
for folder in ['provenance','benchmarks','evidence']:
    records += sorted((root/folder).glob('*.json'))
for path in records:
    data=load(path)
    if path.name == 'environment-local.json':
        required_environment = {'os', 'machine', 'python'}
        if not required_environment.issubset(data): raise SystemExit(f'{path}: incomplete environment record')
        continue
    if data.get('schema_version')!='1.0.0': raise SystemExit(f'{path}: unsupported schema_version')
competitor=load(root/'provenance/competitor-manifest.json')
allowed={'Self-reported','Independently reproduced','Contradicted','Inconclusive','Not comparable','Resolved upstream'}
for artifact in competitor['artifacts']:
    missing={'id','name','category','classification'}-artifact.keys()
    if missing or artifact['classification'] not in allowed: raise SystemExit('invalid competitor manifest entry')
research=load(root/'provenance/research-records.json')
for record in research['records']:
    if record.get('status') not in {'approved','proposed','blocked'}: raise SystemExit('invalid research status')
required=['LICENSE','NOTICE','PROVENANCE.md','KNOWN_FAILURES.md','VERSION','CMakeLists.txt','scripts/verify-release.sh']
missing=[item for item in required if not (root/item).is_file()]
if missing: raise SystemExit('missing required files: '+', '.join(missing))
print('JSON schema structure, record semantics, and required-file checks passed')
