const {chromium}=require('C:/Users/Hello/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('fs'),path=require('path'),assert=require('assert'),crypto=require('crypto');
(async()=>{
 const dir=path.resolve('result/opening_execution'),file=path.join(dir,'短观察期因子初筛.html');
 const data=JSON.parse(fs.readFileSync(path.join(dir,'短观察期_看板数据.json'),'utf8'));
 const browser=await chromium.launch({channel:'msedge',headless:true});const context=await browser.newContext({offline:true,viewport:{width:1500,height:1050}}),page=await context.newPage();const errors=[],external=[];
 page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>{if(/^https?:/.test(r.url()))external.push(r.url())});
 await page.goto(require('url').pathToFileURL(file).href);await page.waitForSelector('body[data-ready="true"]');
 assert.equal(await page.locator('#rows tr').count(),28);let cells=0;
 for(const anchor of ['decision','arrival'])for(const pairs of ['own','diagonal_common']){
   await page.selectOption('#anchor',anchor);await page.selectOption('#pairs',pairs);
   for(const g of data.groups)for(let w=1;w<=5;w++){
     const r=data.scores.find(r=>r.expression===g.expression&&r.w===w&&r.anchor===anchor&&r.pair_set===pairs&&r.label_type==='cumulative'),v=r.raw_score;
     assert.equal(await page.locator(`tr[data-factor="${g.expression}"] td[data-w="${w}"]`).innerText(),v==null?'—':(v>0?'+':'')+v.toFixed(3));cells++;
   }
 }
 await page.selectOption('#anchor','decision');await page.selectOption('#pairs','own');await page.selectOption('#scope','candidate');
 assert.equal(await page.locator('#rows tr').count(),5);assert.equal(await page.locator('tr[data-factor="F07_QuotePosition"]').count(),0);
 await page.screenshot({path:path.join(dir,'网页验收_短观察期候选.png')});
 await page.click('tr[data-factor="F04_QuoteShift"]');assert.equal(await page.locator('#detail svg').count(),2);assert.equal(await page.locator('#detail circle').count(),300);
 assert((await page.locator('#detail').innerText()).includes('第 2–5 秒新增'));
 await page.locator('#detail').screenshot({path:path.join(dir,'网页验收_短观察期曲线.png')});
 await page.selectOption('#anchor','arrival');assert((await page.locator('#detail').innerText()).includes('执行延迟后'));
 await page.selectOption('#scope','all');await page.fill('#search','A04_QISlope');assert.equal(await page.locator('#rows tr').count(),1);assert.equal(await page.locator('#rows td[data-w="1"]').innerText(),'—');
 assert.deepEqual(errors,[]);assert.deepEqual(external,[]);
 const result={html_sha256:crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex'),offline:true,browser_errors:errors,external_requests:external.length,all_table_cells_verified:cells,candidate_rows:5,curve_points:300,missing_slope_shown:true};
 fs.writeFileSync(path.join(dir,'short_window_dashboard_verification.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result));await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
