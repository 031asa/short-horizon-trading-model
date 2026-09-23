"""Package reviewed research outputs, keeping experiments and originals intact."""
from pathlib import Path
import csv
import hashlib
import io
import json
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parents[1]
RESULT=ROOT/'result'
ARCHIVE=RESULT/'研究结果汇总.zip'
PREFIX='短时交易模型研究结果/'
EXCLUSIONS={
    'result/opening_execution/summary_ic.csv':'与summary_ic.parquet重复的超大CSV导出；保留Parquet汇总',
    'result/opening_execution/daily_ic.parquet':'完整逐日IC约1GB，保留在原目录；本包保留完整IC汇总及短窗口逐日IC',
    'result/opening_execution/短观察期_看板数据.json':'已嵌入同目录短观察期因子初筛.html，避免重复打包',
}


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def choose_files():
    files=[]
    omitted=[]
    for folder in ['opening_execution','opening_prediction']:
        for p in sorted((RESULT/folder).rglob('*')):
            if not p.is_file():
                continue
            rel=p.relative_to(ROOT).as_posix()
            if p.is_symlink():
                raise ValueError(f'Unexpected symbolic link: {rel}')
            reason=EXCLUSIONS.get(rel)
            if any(part.startswith('.') for part in p.relative_to(RESULT).parts):
                reason='中间暂存文件'
            elif 'date_validation' in p.parts:
                reason='此前已取消的日期验证实验，不混入已交付结果'
            elif p.name.startswith('fold'):
                reason='分折缓存；已保留合并预测、模型参数和选择过程'
            if reason:
                omitted.append(dict(path=rel,bytes=p.stat().st_size,reason=reason))
            else:
                files.append(p)
    for rel in ['README.md','撮合规则与研究说明.md','项目口径核对.md',
                '研究资料/开盘首分钟_L1特征汇总_双类排版版.pdf',
                'sample_snapshot_原子执行总表.parquet']:
        path=ROOT/rel
        if path.is_file():
            files.append(path)
    if len(files)!=len(set(files)):
        raise ValueError('Duplicate input paths')
    return sorted(files),omitted


def guide(commit,files,omitted):
    return f'''# 短时交易模型：研究结果阅读指南

本包汇总当前已交付的研究结果，源代码版本为 `{commit}`。
解压后保留目录结构。HTML可用浏览器离线打开；CSV为UTF-8编码；Parquet是完整精度的数据文件。

## 建议先看

1. [最新两阶段执行消融报告](result/opening_execution/two_stage_ablation/两阶段执行消融报告.md)：A基准、B先观察、C先挂单；仅第3秒判断、统一第10秒兜底。
2. [执行成本图片](result/opening_execution/two_stage_ablation/两阶段执行成本对比.png) 与 [策略总表](result/opening_execution/two_stage_ablation/策略总表.csv)。
3. [固定三因子增强报告](result/opening_prediction/three_factor_enhancement/固定三因子增强报告.md)：固定三因子、额外信息及1–10秒观察窗口。
4. [开盘时间曲线](result/opening_prediction/three_factor_enhancement/time_of_minute/开盘首分钟_准确率时间曲线.png)。
5. [短观察期因子初筛](result/opening_execution/短观察期因子初筛.md) 与 [全部因子IC看板](result/opening_execution/全部因子IC与衰减.html)。
6. [每天最开始一笔的执行胜率](result/opening_execution/first_order_ablation/首笔执行结果.md)：按首条实际开盘行情发单，单独报告，不能混入名义09:30:00任务。

## 各目录口径

|目录|内容|使用边界|
|---|---|---|
|result/opening_execution 根目录|59家族IC、预期登记、衰减、短窗初筛及异常说明|原IC实验含10秒冷启动；有方向价变使用LastPrice；不要与无冷启动任务混合|
|result/opening_prediction 根目录|早期预测检验、逐任务概率、模型参数|早期模型实验，供历史对照|
|collinearity / significance|六因子及三因子共线性、显著性诊断|条件诊断，不等同于独立样本筛选|
|no_cold_start|无冷启动小组合、完整期限评价|允许组合变化的旧实验|
|three_factor_enhancement|固定三因子增强、全部期限、逐任务数据、模型参数及选择记录|观察＝回看1–10秒，三因子始终保留；时间曲线在time_of_minute|
|opening_execution/two_stage_ablation|最新两阶段执行、成本、归因、逐任务事件与验收|使用四折固定3秒增强模型；18个检验日；每个任务独立模拟买卖各1手|
|opening_execution/first_order_ablation|只看每天最开始发单任务的执行胜率|首条开盘快照为实际起点，约09:30:00.1～00.5；所有阶段从此起点计算|

## 最新结果的含义

两阶段实验中，买卖各半的平均执行成本为A 1.771bp、B 1.431bp、C 1.853bp；B较A节约0.340bp，C较A节约−0.082bp。
C平均完成更快，但本轮没有显示稳定成本改善。成本以同一任务起点LastPrice衡量。
该实验有1,062个共同任务；每天第0秒缺少初始报价而统一排除，不借未来报价填补。
预测结果仍是已有日期上的探索；L1代理撮合不包含排队、手续费、冲击或部分成交。

## 数据完整性与排除项

共打包 {len(files)} 个原始结果／参考文件。`文件清单.csv` 记录每个文件的相对路径、字节数、SHA256。
`打包清单.json` 同时记录未入包文件及原因。压缩包逐文件回读核验内容校验值。

原始全日行情、依赖缓存、暂存目录、分折中间缓存、此前已取消的日期验证实验不入包。
超大逐日IC `result/opening_execution/daily_ic.parquet` 保留在原工作区；本包有完整IC汇总 `summary_ic.parquet`、短窗口逐日IC以及模型和执行实验的逐任务数据。
同内容的约1GB `summary_ic.csv` 不重复打包；需要CSV时可从入包的Parquet导出。
因此这是研究结果交付包，不是包含原始行情和全部中间文件的完整环境备份。

源代码使用Git单独管理；其他机器复跑还需原始行情与项目依赖，见README。
'''


def main():
    RESULT.mkdir(exist_ok=True)
    files,omitted=choose_files()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    manifest=dict(source_commit=commit,files=[],excluded=omitted,
                  excluded_raw_market_data=True,retained_path_structure=True)
    for p in files:
        manifest['files'].append(dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=digest(p)))
    readme=guide(commit,files,omitted)
    buffer=io.StringIO(newline='')
    writer=csv.DictWriter(buffer,fieldnames=['path','bytes','sha256'])
    writer.writeheader();writer.writerows(manifest['files'])
    metadata=json.dumps(manifest,ensure_ascii=False,indent=2)
    temporary=ARCHIVE.with_suffix('.zip.part')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as archive:
        archive.writestr(PREFIX+'阅读指南.md',readme)
        archive.writestr(PREFIX+'文件清单.csv','\ufeff'+buffer.getvalue())
        archive.writestr(PREFIX+'打包清单.json',metadata)
        for p in files:
            archive.write(p,PREFIX+p.relative_to(ROOT).as_posix())
    # Read all compressed entries back; check CRC and cryptographic digest,
    # without extracting or modifying the original research files.
    with zipfile.ZipFile(temporary) as archive:
        assert len(archive.namelist())==len(files)+3
        assert len(set(archive.namelist()))==len(archive.namelist())
        for entry in manifest['files']:
            with archive.open(PREFIX+entry['path']) as stream:
                check=hashlib.file_digest(stream,'sha256').hexdigest()
            assert check==entry['sha256'],entry['path']
        assert archive.read(PREFIX+'阅读指南.md').decode('utf-8')==readme
        assert json.loads(archive.read(PREFIX+'打包清单.json'))==manifest
    temporary.replace(ARCHIVE)
    (RESULT/'研究结果打包说明.md').write_text(readme,encoding='utf-8')
    (RESULT/'研究结果文件清单.csv').write_text('\ufeff'+buffer.getvalue(),encoding='utf-8')
    (RESULT/'研究结果打包清单.json').write_text(metadata,encoding='utf-8')
    verification=dict(archive=ARCHIVE.name,source_commit=commit,files=len(files),
        bytes=ARCHIVE.stat().st_size,sha256=digest(ARCHIVE),all_file_hashes_verified=True,
        uncompressed_bytes=sum(x['bytes'] for x in manifest['files']))
    (RESULT/'研究结果打包验收.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(verification,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
