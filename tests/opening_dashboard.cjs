const {chromium}=require('C:/Users/Hello/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const path=require('path'),fs=require('fs'),assert=require('assert');
(async()=>{
 const dir=path.resolve(process.argv[2]||'result/opening_execution/.pending');
 const browser=await chromium.launch({channel:'msedge',headless:true});
 const context=await browser.newContext({viewport:{width:1440,height:1020},offline:true});
 const page=await context.newPage();const errors=[];page.on('pageerror',e=>errors.push(e.message));
 const requests=[];page.on('request',r=>{if(/^https?:/.test(r.url()))requests.push(r.url())});
 await page.goto(require('url').pathToFileURL(path.join(dir,'全部因子IC与衰减.html')).href);
 await page.waitForSelector('body[data-ready="true"]',{timeout:180000});
 const count=await page.locator('#count').innerText();assert(count.includes('个输出'));
 await page.screenshot({path:path.join(dir,'网页验收_主表.png')});
 let windowInputChecks=0;
 const windowFile=path.join(dir,'window_input_expected.json');
 if(fs.existsSync(windowFile)){
   const expected=JSON.parse(fs.readFileSync(windowFile,'utf8'));
   assert.equal(await page.locator('#windowDefault').getAttribute('type'),'number');
   assert.equal(await page.locator('#windowMode').inputValue(),'grouped');
   await page.click('[data-tab="computed"]');await page.selectOption('#family','F05');await page.fill('#search','Momentum');
   for(const probe of expected){
     await page.fill('#windowDefault',String(probe.h));await page.locator('#windowDefault').press('Enter');
     assert.equal(await page.locator('#table tr[data-factor]').count(),1);
     const row=page.locator('tr[data-factor="'+probe.factor+'"]');
     assert.equal(await row.locator('[data-h="5"]').innerText(),probe.display);
     await row.locator('[data-h="5"]').click();
     assert((await page.locator('#detail').innerText()).includes(probe.factor));
     windowInputChecks++;
   }
   await page.fill('#table .row-window','3');await page.locator('#table .row-window').press('Enter');
   assert.equal(await page.locator('tr[data-factor="F05_Momentum_h3s"] [data-h="5"]').innerText(),expected.find(p=>p.h===3).display);
   assert((await page.locator('#detail').innerText()).includes('F05_Momentum_h3s'));
   await page.fill('#windowDefault','4');await page.locator('#windowDefault').press('Enter');
   assert.equal(await page.locator('tr[data-factor="F05_Momentum_h4s"]').count(),1);
   for(const invalid of ['0','11','2.5','']){
     await page.fill('#windowDefault',invalid);await page.locator('#windowDefault').press('Enter');
     assert((await page.locator('#windowMessage').innerText()).includes('未应用'));
     assert.equal(await page.locator('#windowDefault').inputValue(),'4');
     assert.equal(await page.locator('tr[data-factor="F05_Momentum_h4s"]').count(),1);windowInputChecks++;
   }
   await page.selectOption('#family','B09');await page.fill('#search','BidRecovery');
   assert.equal(await page.locator('tr[data-factor="B09_BidRecovery_h10s"]').count(),1);
   assert.equal(await page.locator('#table .row-window').count(),0);
   await page.selectOption('#priorView','original');assert.equal(await page.locator('#windowDefault').inputValue(),'5');
   await page.selectOption('#priorView','review');await page.fill('#search','');await page.selectOption('#family','all');
   await page.screenshot({path:path.join(dir,'网页验收_可输入窗口.png')});
 }
 let segmentChecks=0;
 for(const [family,name] of [['A08','OFIRateChange'],['C08','MomentumSpeedChange'],['D03','VolumeRateChange'],['D03','VolumeRateImbalance'],['R05','DepthChange'],['R05','RVChange'],['R05','QIVolumeAcceleration']]){
   await page.click('[data-tab="computed"]');await page.selectOption('#family',family);await page.fill('#search',name);
   assert.equal(await page.locator('#table tr[data-factor]').count(),1);
   assert((await page.locator('#table').innerText()).includes('每段长度'));
   for(const length of [2,5]){
     await page.fill('#table .row-window',String(length));await page.locator('#table .row-window').press('Enter');
     const factor=await page.locator('#table tr[data-factor]').getAttribute('data-factor');assert(factor.endsWith('_lag'+length+'s'));
     const values=await page.locator('#table td[data-h]').allTextContents();
     await page.selectOption('#windowMode','expanded');
     assert.deepEqual(await page.locator('tr[data-factor="'+factor+'"] td[data-h]').allTextContents(),values);
     await page.selectOption('#windowMode','grouped');segmentChecks++;
   }
   await page.fill('#windowDefault','3');await page.locator('#windowDefault').press('Enter');
   assert.equal(await page.locator('#table .row-window').inputValue(),'5');
   await page.fill('#table .row-window','3');await page.locator('#table .row-window').press('Enter');
   assert.equal(await page.locator('#table .row-window').inputValue(),'5');assert((await page.locator('#windowMessage').innerText()).includes('未应用'));
 }
 await page.selectOption('#family','C08');await page.fill('#search','MomentumSpeedChange');
 await page.screenshot({path:path.join(dir,'网页验收_分段合并.png')});
 await page.fill('#windowDefault','5');await page.locator('#windowDefault').press('Enter');
 await page.fill('#search','');await page.selectOption('#family','all');
 // Expanded mode still exposes every physical version for legacy probes and exports.
 await page.selectOption('#windowMode','expanded');await page.click('[data-tab="all"]');
 const probes=JSON.parse(fs.readFileSync(path.join(dir,'dashboard_qa_probes.json'),'utf8'));let verified=0;
 for(const probe of probes){
   await page.selectOption('#family',probe.family_id);await page.selectOption('#history','all');
   await page.selectOption('#obs',String(probe.observation_seconds));await page.selectOption('#anchor',probe.anchor);
   await page.selectOption('#target',probe.target);await page.selectOption('#pairs',probe.pair_set);
   await page.selectOption('#kind',probe.label_type);await page.selectOption('#method',probe.method);
   await page.fill('#search',probe.factor);
   const formula=await page.locator('#icFormula').innerText();
   assert(formula.includes(probe.method==='ic'?'Pearson IC':'Spearman Rank IC'));
   assert.equal(formula.includes('Rank(X'),probe.method==='rank_ic');
   assert.equal(formula.includes('(u−1)'),probe.label_type==='incremental');
   assert(formula.includes(probe.target==='absolute'?'绝对':'有方向'));
   assert(formula.includes('+ '+probe.observation_seconds+'s'));
   assert(formula.includes(probe.anchor==='arrival'?'观察结束后第 2 张快照':'a = t_dec'));
   assert(formula.includes(probe.pair_set==='own'?'同时有效的任务':'全部 30 个期限均有效的共同任务'));
   assert(formula.includes('日期等权，不按配对数加权'));

   const row=page.locator('tr[data-factor="'+probe.factor+'"]');
   const cell=row.locator('td[data-h="'+probe.horizon_seconds+'"]');
   const shown=await cell.innerText();assert.equal(shown,probe.display,JSON.stringify(probe));
   await cell.click();await page.selectOption('#detailH',String(probe.horizon_seconds));
   const detail=await page.locator('#detail').innerText();assert(detail.includes(probe.factor));
   assert(detail.includes(probe.meanDisplay));assert(await page.locator('#detail svg').count()===2);
   verified++;
 }
 await page.selectOption('#family','F01');await page.fill('#search','');await page.selectOption('#obs','1');await page.selectOption('#anchor','decision');await page.selectOption('#target','signed');await page.selectOption('#pairs','own');await page.selectOption('#kind','cumulative');await page.selectOption('#method','rank_ic');
 await page.locator('tr[data-factor="F01_QI"] td[data-h="5"]').click();
 await page.screenshot({path:path.join(dir,'网页验收_明细.png')});
 // Check all 30 curve points against the packed source arrays via rendered SVG titles.
 const titles=await page.locator('#detail .chart').first().locator('circle title').allTextContents();
 assert.equal(titles.length,30);assert(titles[0].startsWith('1s:'));assert(titles[29].startsWith('30s:'));
 await page.selectOption('#family','B09');await page.fill('#search','B09_BidRecovery_h10s');
 const recovery=page.locator('tr[data-factor="B09_BidRecovery_h10s"]');assert.equal(await recovery.locator('td[data-h="1"]').innerText(),'—');await recovery.click();assert((await page.locator('#detail').innerText()).includes('事件或支持数不足'));
 await page.click('#closeDetail');await page.fill('#search','');await page.selectOption('#family','R06');await page.click('[data-tab="quality"]');assert((await page.locator('#table').innerText()).includes('质量字段，不计算方向 IC'));
 await page.selectOption('#family','all');await page.click('[data-tab="families"]');assert.equal(await page.locator('#table tbody tr').count(),59);
 await page.click('[data-family="R05"]');await page.click('[data-tab="alias"]');assert.equal(await page.locator('#table tbody tr').count(),6);
 await page.click('[data-tab="all"]');await page.selectOption('#family','all');assert((await page.locator('#count').innerText()).includes('个输出'));await page.click('#next');assert((await page.locator('#page').innerText()).startsWith('2 /'));
 // Check the complete cross-page ordering for both evaluation targets.
 const registry=await page.locator('#info').evaluate(e=>JSON.parse(e.textContent).registry);
 const families=await page.locator('#info').evaluate(e=>Object.keys(JSON.parse(e.textContent).families));
 await page.click('[data-tab="all"]');await page.selectOption('#family','all');await page.selectOption('#history','all');await page.fill('#search','');await page.selectOption('#signFilter','all');
 const reviewed=registry.some(r=>r.review_version);
 const priorViews=reviewed?['review','original']:['original'];
 for(const view of priorViews)for(const target of ['signed','absolute']){
   if(reviewed)await page.selectOption('#priorView',view);
   await page.selectOption('#target',target);
   const actual=[];
   for(;;){actual.push(...await page.locator('#table tr[data-factor]').evaluateAll(rows=>rows.map(r=>r.dataset.factor)));if(await page.locator('#next').isDisabled())break;await page.click('#next')}
   assert.equal(actual.length,registry.filter(r=>view!=='original'||(r.review_origin!=='derived'&&r.window_extension!==true)).length);
   const byName=new Map(registry.map(r=>[r.factor,r]));const order={positive:0,negative:1,uncertain:2,not_applicable:3};
   for(let i=1;i<actual.length;i++){
     const key=(view==='original'&&reviewed?'original_':'')+'expected_sign_'+target;
     const a=byName.get(actual[i-1]),b=byName.get(actual[i]),pa=order[a[key]],pb=order[b[key]];
     assert(pa<=pb,'Logical prior groups out of order');
     if(pa===pb)assert(families.indexOf(a.family_id)<=families.indexOf(b.family_id),'Family order changed inside logical prior group');
   }
 }
 // Compare the new positive-prior/green-IC filter with full-precision summary selections.
 const filterCases=JSON.parse(fs.readFileSync(path.join(dir,'positive_filter_expected.json'),'utf8'));
 await page.selectOption('#family','all');await page.selectOption('#history','all');await page.fill('#search','');
 await page.selectOption('#obs','1');await page.selectOption('#anchor','decision');await page.selectOption('#pairs','own');await page.selectOption('#kind','cumulative');await page.click('[data-tab="computed"]');
 await page.selectOption('#signFilter','positive_green');
 for(const test of filterCases){
   if(reviewed)await page.selectOption('#priorView',test.view);
   await page.selectOption('#target',test.target);await page.selectOption('#method',test.method);await page.selectOption('#greenH',test.scope);
   const actual=[];
   for(;;){actual.push(...await page.locator('#table tr[data-factor]').evaluateAll(rows=>rows.map(r=>r.dataset.factor)));if(await page.locator('#next').isDisabled())break;await page.click('#next')}
   assert.deepEqual(actual,test.factors,JSON.stringify(test));
 }
 if(reviewed)await page.selectOption('#priorView','review');
 await page.selectOption('#target','signed');await page.selectOption('#method','rank_ic');await page.selectOption('#greenH','any');
 const defaultCount=filterCases.find(x=>x.view===(reviewed?'review':'original')&&x.target==='signed'&&x.method==='rank_ic'&&x.scope==='any').factors.length;
 assert((await page.locator('#count').innerText()).startsWith(defaultCount+' '));
 await page.screenshot({path:path.join(dir,'网页验收_正向绿色筛选.png')});
 await page.selectOption('#signFilter','all');
 await page.goto(require('url').pathToFileURL(path.join(dir,'全部因子IC与衰减.html')).href+'#positive');
 assert.equal(await page.locator('#signFilter').inputValue(),'positive_green');
 assert((await page.locator('#count').innerText()).startsWith(defaultCount+' '));
 if(reviewed){
   await page.selectOption('#signFilter','all');await page.selectOption('#origin','derived');
   assert((await page.locator('#count').innerText()).startsWith(registry.filter(r=>r.kind==='computed'&&r.review_origin==='derived').length+' '));
   await page.fill('#search','M01_CounterflowDeviation_h5s');
   const newRow=page.locator('tr[data-factor="M01_CounterflowDeviation_h5s"]');
   assert.equal(await newRow.locator('.pin3').innerText(),'负向');await newRow.click();
   assert((await page.locator('#detail').innerText()).includes('父因子'));
   assert((await page.locator('#detail').innerText()).includes('相反机制'));
   await page.selectOption('#origin','original');await page.fill('#search','M01_MADeviation_h5s');
   const originalRow=page.locator('tr[data-factor="M01_MADeviation_h5s"]');
   const oldIC=await originalRow.locator('[data-h="1"]').innerText();
   assert.equal(await originalRow.locator('.pin3').innerText(),'负向');
   await page.selectOption('#priorView','original');
   assert.equal(await originalRow.locator('.pin3').innerText(),'不确定');
   assert.equal(await originalRow.locator('[data-h="1"]').innerText(),oldIC);
   assert((await page.locator('#reviewNote').innerText()).includes('不确定 221'));
   await page.selectOption('#priorView','review');await page.selectOption('#origin','all');
   assert((await page.locator('#reviewNote').innerText()).includes('不确定 '+registry.filter(r=>r.kind==='computed'&&r.expected_sign_signed==='uncertain').length));
   await page.fill('#search','');await page.selectOption('#family','M01');
   await page.screenshot({path:path.join(dir,'网页验收_逻辑预期审阅.png')});
 }
 assert.deepEqual(errors,[]);assert.deepEqual(requests,[]);
 const result={html_sha256:require('crypto').createHash('sha256').update(fs.readFileSync(path.join(dir,'全部因子IC与衰减.html'))).digest('hex'),offline_file_open:true,external_requests:requests.length,browser_errors:errors,sort_targets_verified:2*priorViews.length,formula_option_probes:probes.length,positive_filter_cases:filterCases.length,positive_filter_default_count:defaultCount,summary_cell_probes:verified,curve_points:titles.length,families:59,window_input_checks:windowInputChecks,segment_parameter_checks:segmentChecks,review_origin_and_old_prior_view:reviewed?'passed':'not applicable',filters_detail_alias_quality_pagination:'passed'};
 fs.writeFileSync(path.join(dir,'dashboard_verification.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result));await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
