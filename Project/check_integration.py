import urllib.request

with urllib.request.urlopen('http://localhost:8000/') as r:
    html = r.read().decode('utf-8', errors='replace')
    print('HTTP Status :', r.status)
    print('Content-Type:', r.headers.get('Content-Type'))

checks = [
    ('smoteCountsText ID',        'id="smoteCountsText"'),
    ('metric-lr-acc ID',          'id="metric-lr-acc"'),
    ('metric-rf-f1 ID',           'id="metric-rf-f1"'),
    ('metric-xgboost-auc ID',     'id="metric-xgboost-auc"'),
    ('resultsSubtitle ID',        'id="resultsSubtitle"'),
    ('modelInfoRow ID',           'id="modelInfoRow"'),
    ('loanDetailsSection ID',     'id="loanDetailsSection"'),
    ('loanDetailsGrid ID',        'id="loanDetailsGrid"'),
    ('fetch /api/assess',         '/api/assess'),
    ('loadMetrics function',      'async function loadMetrics'),
    ('DOMContentLoaded hook',     'DOMContentLoaded'),
    ('Financial Analysis label',  'Financial Analysis'),
    ('Affordability rendering',   'affordabilityStatus'),
]

all_ok = True
for label, token in checks:
    found = token in html
    status = 'OK     ' if found else '** MISSING **'
    if not found:
        all_ok = False
    print(f'  [{status}]  {label}')

print()
print('All checks passed!' if all_ok else 'SOME CHECKS FAILED - see above.')
