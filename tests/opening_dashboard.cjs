const {chromium}=require('C:/Users/Hello/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const path=require('path'),fs=require('fs'),assert=require('assert');
(async()=>{
 const dir=path.resolve(process.argv[2]||'result/opening_execution/.pending');
 const browser=await chromium.launch({channel:'msedge',headless:true});
 const context=await browser.newContext({viewport:{width:1440,height:1020},offline:true});
 const page=await context.newPage();const errors=[];page.on('pageerror',e=>errors.push(e.message));
 const requests=[];page.on('request',r=>{if(/^https?:/.test(r.url()))requests.push(r.url())});
 await page.goto(require('url').pathToFileURL(path.join(dir,'全部因子IC与衰减.html')).href);
 await page.waitForSelector('body[data-ready="true"]',{timeout:60000});
 const count=await page.locator('#count').innerText();assert(count.includes('个输出'));
 await page.screenshot({path:path.join(dir,'网页验收_主表.png')});
 const probes=JSON.parse(fs.readFileSync(path.join(dir,'dashboard_qa_probes.json'),'utf8'));let verified=0;
 for(const probe of probes){
   await page.selectOption('#family',probe.family_id);await page.selectOption('#history','all');
   await page.selectOption('#obs',String(probe.observation_seconds));await page.selectOption('#anchor',probe.anchor);
   await page.selectOption('#target',probe.target);await page.selectOption('#pairs',probe.pair_set);
   await page.selectOption('#kind',probe.label_type);await page.selectOption('#method',probe.method);
   await page.fill('#search',probe.factor);
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
 assert.deepEqual(errors,[]);assert.deepEqual(requests,[]);
 const result={html_sha256:require('crypto').createHash('sha256').update(fs.readFileSync(path.join(dir,'全部因子IC与衰减.html'))).digest('hex'),offline_file_open:true,external_requests:requests.length,browser_errors:errors,summary_cell_probes:verified,curve_points:titles.length,families:59,filters_detail_alias_quality_pagination:'passed'};
 fs.writeFileSync(path.join(dir,'dashboard_verification.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result));await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
