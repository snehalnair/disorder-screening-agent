"""Download result files pushed to GitHub by the Colab notebook."""
import base64, json, os, pathlib, sys
import urllib.request, urllib.error

REPO   = 'snehalnair/disorder-screening-agent'
FILES  = [
    'evaluation/results/rq2_disorder_novel15.json',
    'evaluation/results/rq2_disorder_all23.json',
]

ROOT = pathlib.Path(__file__).parent.parent

token = os.environ.get('GITHUB_TOKEN')
if not token:
    print('ERROR: set GITHUB_TOKEN environment variable first.')
    print('  export GITHUB_TOKEN=your_token')
    sys.exit(1)

for github_path in FILES:
    url = f'https://api.github.com/repos/{REPO}/contents/{github_path}'
    req = urllib.request.Request(url, headers={
        'Authorization': f'token {token}',
        'Accept': 'application/vnd.github.v3+json',
    })
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())
        content = base64.b64decode(data['content'])
        dest = ROOT / github_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        print(f'✓  {github_path}  ({len(content):,} bytes)')
    except urllib.error.HTTPError as e:
        print(f'✗  {github_path}  HTTP {e.code}: {e.reason}')

print('\nDone. Now run:')
print('  python -m evaluation.eval_accuracy \\')
print('      --results evaluation/results/rq2_disorder_all23.json \\')
print('      --save evaluation/results/rq3_accuracy_all23.json')
