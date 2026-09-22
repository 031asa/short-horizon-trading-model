"""Update current prior metadata and offline UI without recomputing any IC."""
from pathlib import Path
import sys,json,hashlib,re
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'.cache/python-packages'))
import pandas as pd
from utils.directional_review import conservative_registry,CONSERVATIVE_VERSION
from scripts.render_opening_ic import refresh_dashboard_template
from scripts.filter_positive_ic import filter_positive

def main():
    out=ROOT/'result/opening_execution'
    registry=conservative_registry(pd.read_csv(out/'feature_registry.csv'))
    registry.to_csv(out/'feature_registry.csv',index=False,encoding='utf-8-sig')
    coverage=pd.read_csv(out/'factor_coverage.csv').set_index('factor')
    for col in registry.columns:
        if col!='factor':coverage[col]=registry.set_index('factor')[col]
    coverage=coverage.reset_index()
    coverage.to_csv(out/'factor_coverage.csv',index=False,encoding='utf-8-sig')
    changed=registry.loc[registry.review_change.eq('assumption_withdrawn')]
    changed.to_csv(out/'主假设撤回清单.csv',index=False,encoding='utf-8-sig')
    path=out/'全部因子IC与衰减.html';html=path.read_text(encoding='utf-8')
    tag='<script id="info" type="application/json">'
    old=html.split(tag,1)[1].split('</script>',1)[0];info=json.loads(old)
    before=hashlib.sha256(html.split('<script id="payload"',1)[1].split('</script>',1)[0].encode()).hexdigest()
    records=coverage.fillna('').to_dict('records')
    for r in records:r['status_counts']=json.loads(r['status_counts'])
    info['registry']=records;info['meta']['prior_version']=CONSERVATIVE_VERSION
    path.write_text(html.replace(tag+old,tag+json.dumps(info,ensure_ascii=False)),encoding='utf-8')
    refresh_dashboard_template(out)
    after=hashlib.sha256(path.read_text(encoding='utf-8').split('<script id="payload"',1)[1].split('</script>',1)[0].encode()).hexdigest()
    assert before==after
    def key(r):
        f=r.factor
        if re.match(r'^(A08|C08|D03|R05)_',f) and not re.search(r'_h\d+s',f):f=re.sub(r'_lag(?:2|5)s$','_lag{S}s',f)
        return re.sub(r'_h\d+s(?=_|$)','_h{W}s',f)
    computed=registry.loc[registry.kind.eq('computed')].copy();computed['expression']=computed.apply(key,axis=1)
    groups=computed.drop_duplicates('expression');counts=groups.expected_sign_signed.value_counts().to_dict()
    note=f'当前保守口径：{len(groups)} 种计算表达中，正向 {counts.get("positive",0)}、负向 {counts.get("negative",0)}、不确定 {counts.get("uncertain",0)}。此前额外延续、回归、单侧主导及新增交互主假设撤回；因子值和全部原始 IC 不变。计算前冻结文件保留历史内容，当前预期以 feature_registry.csv 为准。'
    (out/'主假设撤回说明.md').write_text('# 当前逻辑预期修正\n\n'+note+'\n\n[逐项清单](主假设撤回清单.csv)\n',encoding='utf-8')
    p=out/'IC研究结果.md';s=p.read_text(encoding='utf-8');s=re.sub(r'\n当前保守口径：[^\n]*\n','\n',s);p.write_text(s.split('\n',1)[0]+'\n\n'+note+'\n'+s.split('\n',1)[1],encoding='utf-8')
    filter_positive(out)
    print(json.dumps({'counts':counts,'withdrawn_versions':len(changed.loc[changed.kind.eq('computed')]),'embedded_numeric_payload_unchanged':before==after},ensure_ascii=False))

if __name__=='__main__':main()
